import unittest
from types import SimpleNamespace
from unittest.mock import patch

from support import controller_type

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

    def test_epoch_arrival_uses_wall_clock_and_monotonic_uses_monotonic(self):
        cls = controller_type(public=True)
        with (
            patch("time.time", return_value=1_800_000_010.0),
            patch("time.monotonic", return_value=110.0),
        ):
            self.assertEqual(
                cls._starvation_waited_s(SimpleNamespace(arrival_time=1_800_000_000.0)),
                10.0,
            )
            self.assertEqual(
                cls._starvation_waited_s(SimpleNamespace(arrival_time=100.0)), 10.0
            )

    def test_public_wait_queue_ages_without_changing_capacity_ranks(self):
        from support import Queue, request

        cls = controller_type(public=True)
        state = object.__new__(cls)
        state.program_enabled = state.starvation_guard_enabled = True
        state.starvation_guard_threshold_s, state.starvation_guard_quantum_s = 4.0, 1.0
        state._starvation_guard_seen = set()
        state.policy, state.running = "priority", []
        state.order = dict(old=0, new=1)
        state.program_priorities = dict(old=5, new=0)
        older, newer = request("old"), request("new")
        older.arrival_time, newer.arrival_time = 1_800_000_000.0, 1_800_000_010.0
        state.waiting, state.skipped_waiting = Queue([newer, older]), Queue()
        state._emit = lambda *args, **kwargs: None
        with patch("time.time", return_value=1_800_000_010.0):
            state._refresh_priority()
        self.assertEqual(older.priority, 0)
        self.assertEqual(state.program_priorities, dict(old=5, new=0))


if __name__ == "__main__":
    unittest.main()
