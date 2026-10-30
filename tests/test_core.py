import unittest
import tempfile
import torch
from remem.config import Config
from remem.model import ReMem
from remem.losses import compute_loss
from remem.metrics import evaluate_metrics, binary_auc
from remem.memory import chronological_groups
from remem.utils import save_checkpoint, load_checkpoint


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **kwargs):
        torch.manual_seed(7)
        return ReMem(Config(dim=16, vocab_size=128, steps=2, **kwargs))

    def test_groups_preserve_original_indices(self):
        self.assertEqual(chronological_groups(torch.tensor([0, 1, 2, 4, 5]), 2), [[0, 1], [2], [4, 5]])

    def test_partition_and_restoration(self):
        model = self.model(restore_threshold=0.0, kappa=1.0)
        output = model("science", ["science", "sports", "history", "music", "space"])
        previous = set()
        for s in output["steps"]:
            active = set(torch.where(s["active"])[0].tolist())
            restored = set(torch.where(s["restored"])[0].tolist())
            memory = set(i for k, e in enumerate(s["episodes"]) if not s["reopen"][k] for i in e.indices.tolist())
            self.assertFalse(active & restored)
            self.assertFalse(restored & memory)
            self.assertEqual(active | restored | memory, set(range(5)))
            self.assertTrue(previous <= restored)
            self.assertEqual(s["post_length"], 5)
            previous = restored
        self.assertTrue(previous)

    def test_no_restore_and_singleton(self):
        model = self.model(restore_threshold=1.0)
        for hist in [["one"], ["one", "two", "three"]]:
            result = model("target", hist, teacher=True)
            loss, _ = compute_loss(result, 1, model.cfg)
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(any(s["reopen"].any() for s in result["steps"]))

    def test_backprop_all_modules(self):
        model = self.model(restore_threshold=0.0, kappa=1.0)
        result = model("science", ["sports", "music", "science", "history", "space"], teacher=True)
        loss, components = compute_loss(result, 1, model.cfg)
        loss.backward()
        self.assertEqual(set(components), {"task", "pred", "state", "mem", "budget", "rest"})
        for module in [model.router, model.memory, model.restorer, model.head, model.backbone]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_checkpoint_roundtrip(self):
        model = self.model().eval()
        with tempfile.TemporaryDirectory() as d:
            path = d + "/weights.pt"
            save_checkpoint(path, model, model.cfg, 0)
            loaded, _ = load_checkpoint(path, "cpu")
            loaded.eval()
            with torch.no_grad():
                a = model("target", ["a", "b"])["logits"]
                b = loaded("target", ["a", "b"])["logits"]
            torch.testing.assert_close(a, b)

    def test_known_metrics(self):
        self.assertEqual(binary_auc([0, 1], [.5, .5]), .5)
        self.assertEqual(binary_auc([0, 1], [.1, .9]), 1)
        self.assertIsNone(binary_auc([1, 1], [.1, .9]))
        m = evaluate_metrics([0, 1, 0], [.1, .9, .2], ["g"] * 3, k=1)
        self.assertEqual(m["ndcg@1"], 1)
        self.assertEqual(m["recall@1"], 1)

    def test_empty_history_rejected(self):
        with self.assertRaises(ValueError):
            self.model()("target", [])


if __name__ == "__main__":
    unittest.main()
