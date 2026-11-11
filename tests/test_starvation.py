import unittest

from kairos.scheduling.starvation import aged_priority


class StarvationPolicyTests(unittest.TestCase):
    def test_before_threshold_preserves_j_rank(self):
        self.assertEqual(aged_priority(7, 3.99)["priority"], 7)

    def test_credit_grows_one_rank_per_quantum(self):
        self.assertEqual(aged_priority(7, 4.0)["priority"], 6)
        self.assertEqual(aged_priority(7, 5.01)["priority"], 5)

    def test_credit_is_bounded_at_queue_head(self):
        result = aged_priority(7, 100.0)
        self.assertEqual(result["priority"], 0)
        self.assertEqual(result["credit"], 7)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            aged_priority(-1, 1.0)
        with self.assertRaises(ValueError):
            aged_priority(1, 1.0, quantum_s=0.0)


if __name__ == "__main__":
    unittest.main()
