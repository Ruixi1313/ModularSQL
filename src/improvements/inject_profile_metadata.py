#!/usr/bin/env python3
"""
Pattern 2 Module 2.3:
Augment column descriptions in DeepEye-SQL's dataset snapshot with
profile-derived hints. Pure post-processing of the snapshot — no DeepEye
source modification.

After running this script, all downstream stages (value retrieval, schema
linking, SQL generation, revision, selection) will see the enriched
descriptions in their prompts automatically, because they all read
column_schema_dict['description'] via DeepEye's _format_single_table_profile().
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.improvements.profile_hint import generate_profile_hint


PROFILE_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL/workspace/profile_cache"


def load_profiles(profile_dir: Path) -> dict:
    """Return {db_id: {table: {col: profile_dict}}}."""
    profiles = {}
    for f in profile_dir.glob("*.json"):
        prof = json.loads(f.read_text())
        db_id = prof["db_id"]
        col_index = {}
        for table_name, tbl in prof["tables"].items():
            col_index[table_name] = {col: stats for col, stats in tbl["columns"].items()}
        profiles[db_id] = col_index
    return profiles


def augment_item(item: dict, profiles: dict, sep: str = " | ") -> tuple[int, int]:
    """
    Mutate `item` in place. Augments column descriptions with [Profile: ...]
    hints if a hint is available.

    Returns (n_augmented, n_skipped).
    """
    db_id = item.get("input", {}).get("database_id") or item.get("database_id")
    if not db_id or db_id not in profiles:
        return 0, 0
    db_profiles = profiles[db_id]

    schema = item.get("input", {}).get("database_schema") or item.get("database_schema") or {}
    tables = schema.get("tables", {})
    aug = skip = 0
    for table_name, table in tables.items():
        table_profile = db_profiles.get(table_name) or {}
        for col_name, col in table.get("columns", {}).items():
            col_profile = table_profile.get(col_name)
            if not col_profile:
                skip += 1
                continue
            hint = generate_profile_hint(col_profile)
            if not hint:
                skip += 1
                continue
            existing = col.get("description") or ""
            # Strip any prior [Profile: ...] block so re-injection updates rather than accumulates
            import re as _re
            stripped = _re.sub(r"\s*\|?\s*\[Profile:[^\]]*\]\s*", "", existing).rstrip(sep + " ")
            col["description"] = (stripped + sep + hint).strip(sep + " ")
            aug += 1
    return aug, skip


def inject(snapshot_path: Path, profile_dir: Path) -> dict:
    """Modify the snapshot file in place, augmenting all column descriptions."""
    items = [json.loads(line) for line in snapshot_path.read_text().splitlines() if line.strip()]
    profiles = load_profiles(profile_dir)

    total_aug = total_skip = 0
    n_items = 0
    for item in items:
        aug, skip = augment_item(item, profiles)
        total_aug += aug
        total_skip += skip
        n_items += 1

    # Atomic write
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
    tmp.replace(snapshot_path)

    return {
        "items": n_items,
        "columns_augmented": total_aug,
        "columns_skipped": total_skip,
        "snapshot": str(snapshot_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True,
                   help="Path to dataset items.jsonl (e.g., workspace/dataset/bird/dev.snapshot.data/items.jsonl)")
    p.add_argument("--profile-dir", default=str(PROFILE_DIR_DEFAULT))
    p.add_argument("--dry-run", action="store_true",
                   help="Print 5 example augmentations without writing.")
    args = p.parse_args()

    snapshot = Path(args.snapshot)
    profile_dir = Path(args.profile_dir)

    if not snapshot.exists():
        print(f"snapshot not found: {snapshot}", file=sys.stderr)
        sys.exit(1)
    if not profile_dir.exists():
        print(f"profile dir not found: {profile_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        items = [json.loads(line) for line in snapshot.read_text().splitlines() if line.strip()]
        profiles = load_profiles(profile_dir)
        for item in items[:1]:
            db = item["input"]["database_id"]
            schema = item["input"]["database_schema"]
            print(f"=== Dry-run on first item (db={db}) ===\n")
            for tbl_name, tbl in list(schema["tables"].items())[:1]:
                for col_name, col in list(tbl["columns"].items())[:5]:
                    before = col.get("description", "")
                    cp = profiles.get(db, {}).get(tbl_name, {}).get(col_name)
                    hint = generate_profile_hint(cp) if cp else ""
                    after = (before + " | " + hint).strip(" |") if hint else before
                    print(f"[{tbl_name}.{col_name}]")
                    print(f"  before: {before[:120]}")
                    print(f"  hint:   {hint}")
                    print(f"  after:  {after[:200]}")
                    print()
        return

    stats = inject(snapshot, profile_dir)
    print(f"✓ Augmented {stats['columns_augmented']} columns across {stats['items']} items "
          f"(skipped {stats['columns_skipped']})")
    print(f"  Snapshot: {stats['snapshot']}")


if __name__ == "__main__":
    main()
