import csv
import tempfile
import unittest
from prepare_data import convert


class DataTests(unittest.TestCase):
    def test_no_future_or_same_time_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            with open(d + "/events.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["user_id", "item_id", "timestamp", "text", "label"])
                writer.writeheader()
                for time in range(20):
                    for label in [0, 1]:
                        writer.writerow(dict(user_id="u", item_id=f"{time}-{label}", timestamp=time,
                                             text=f"event-{time}-{label}", label=label))
            splits = convert(d + "/events.csv", d + "/out")
            for rows in splits.values():
                self.assertTrue(rows)
                for row in rows:
                    t = int(row["target"].split("-")[1])
                    self.assertTrue(all(int(x.split("-")[1]) < t for x in row["history"]))
                    self.assertTrue(all(x.endswith("-1") for x in row["history"]))
