"""Run with: python3 -m unittest discover -s examples/multiplicity -v"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

from demo import compare_rows, main, run_demo


class MultiplicityTests(unittest.TestCase):
    def test_distinct_loses_required_duplicates(self):
        case = run_demo()[0]
        self.assertEqual(case["reference"], [("Ada",), ("Ada",), ("Bo",)])
        self.assertEqual(case["candidate"], [("Ada",), ("Bo",)])
        self.assertTrue(case["set_equal"])
        self.assertFalse(case["bag_equal"])

    def test_missing_distinct_keeps_unwanted_duplicates(self):
        case = run_demo()[1]
        self.assertEqual(case["reference"], [("Ada",), ("Bo",)])
        self.assertEqual(case["candidate"], [("Ada",), ("Ada",), ("Bo",)])
        self.assertTrue(case["set_equal"])
        self.assertFalse(case["bag_equal"])

    def test_order_is_ignored(self):
        rows = [("Ada",), ("Ada",), ("Bo",)]
        self.assertEqual(compare_rows(rows, list(reversed(rows))),
                         {"set_equal": True, "bag_equal": True})

    def test_different_values_fail_both(self):
        self.assertEqual(compare_rows([("Ada",)], [("Bo",)]),
                         {"set_equal": False, "bag_equal": False})

    def test_equal_lengths_do_not_imply_bag_equality(self):
        left = [("Ada",), ("Ada",), ("Bo",)]
        right = [("Ada",), ("Bo",), ("Bo",)]
        self.assertEqual(compare_rows(left, right),
                         {"set_equal": True, "bag_equal": False})

    def test_empty_results(self):
        self.assertEqual(compare_rows([], []),
                         {"set_equal": True, "bag_equal": True})
        self.assertEqual(compare_rows([], [("Ada",)]),
                         {"set_equal": False, "bag_equal": False})

    def test_entire_rows_and_column_positions_matter(self):
        self.assertEqual(compare_rows([("Ada", "Bo")], [("Bo", "Ada")]),
                         {"set_equal": False, "bag_equal": False})

    def test_each_run_has_a_fresh_database(self):
        self.assertEqual(run_demo(), run_demo())

    def test_expected_output(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main()
        expected = Path(__file__).with_name("expected_output.txt").read_text()
        self.assertEqual(output.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
