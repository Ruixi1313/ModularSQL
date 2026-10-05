"""Synthetic SQLite illustration of set versus bag result comparison.

This is an educational example, not the ModularSQL runtime guardrail.
Only Python's standard library is used; the database lives in memory.
"""

from collections import Counter
from contextlib import closing
import sqlite3


SCHEMA = """
CREATE TABLE purchases (purchase_id INTEGER PRIMARY KEY, customer TEXT NOT NULL);
INSERT INTO purchases VALUES (1, 'Ada'), (2, 'Ada'), (3, 'Bo');
"""
ALL_PURCHASES = "SELECT customer FROM purchases ORDER BY customer, purchase_id"
UNIQUE_CUSTOMERS = "SELECT DISTINCT customer FROM purchases ORDER BY customer"


def compare_rows(candidate, reference):
    """Compare materialized row tuples, ignoring order but preserving columns.

    Both inputs must be successful query results, never error sentinels.
    Python tuple equality is sufficient for this example's TEXT values; this
    is not a general SQL-dialect/type/collation equivalence checker.
    """
    return {
        "set_equal": set(candidate) == set(reference),
        "bag_equal": Counter(candidate) == Counter(reference),
    }


def run_demo():
    """Execute both queries on a fresh synthetic database and return two cases."""
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA)
        all_rows = connection.execute(ALL_PURCHASES).fetchall()
        unique_rows = connection.execute(UNIQUE_CUSTOMERS).fetchall()

    # The natural-language task decides whether duplicates are meaningful.
    return [
        {
            "name": "One customer name per purchase (duplicates required)",
            "reference": all_rows,
            "candidate": unique_rows,
            **compare_rows(unique_rows, all_rows),
        },
        {
            "name": "Unique customer names (duplicates unwanted)",
            "reference": unique_rows,
            "candidate": all_rows,
            **compare_rows(all_rows, unique_rows),
        },
    ]


def main():
    for index, case in enumerate(run_demo()):
        if index:
            print()
        print(case["name"])
        print(f"  reference: {case['reference']}")
        print(f"  candidate: {case['candidate']}")
        print(f"  set_equal: {case['set_equal']}")
        print(f"  bag_equal: {case['bag_equal']}")


if __name__ == "__main__":
    main()
