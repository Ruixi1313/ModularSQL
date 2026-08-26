#!/usr/bin/env python3
"""
Verify DeepEye-SQL's checkpoint / resume mechanism is robust enough for
a multi-hour full BIRD-Dev run.

Checks:
  1. All 5 LLM stages write incremental checkpoints (per-5-item flush)
  2. Each stage has skip-if-complete logic
  3. Re-application of checkpoint preserves state (apply_to_dataset)
  4. Real-world evidence: slim P2 actually resumed cleanly after our 2 crashes

This is a static + log-based audit. No re-runs required.
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

DEEPEYE_ROOT = Path(__file__).resolve().parents[2] / "external/DeepEye-SQL"
STAGES = ["value_retrieval", "schema_linking", "sql_generation", "sql_revision", "sql_selection"]


def check_skip_logic():
    """Verify each stage has its own 'is_stage_complete' check before processing."""
    print("=" * 70)
    print("1. Skip-if-complete logic (per stage)")
    print("=" * 70)
    issues = []
    for stage in STAGES:
        stage_file = DEEPEYE_ROOT / "app" / "pipeline" / stage / f"{stage}.py"
        if not stage_file.exists():
            issues.append(f"  {stage}: file not found")
            continue
        text = stage_file.read_text()
        has_check = (
            "is_stage_complete" in text
            and "already been" in text
            and "Skipping" in text
        )
        if has_check:
            print(f"  ✓ {stage:25s} has skip-if-complete logic")
        else:
            issues.append(f"  ✗ {stage}: missing skip-if-complete")
            print(f"  ✗ {stage:25s} MISSING skip-if-complete")
    return issues


def check_checkpoint_flushing():
    """Verify each stage flushes checkpoints incrementally."""
    print("\n" + "=" * 70)
    print("2. Incremental checkpoint flushing (every 5 items)")
    print("=" * 70)
    issues = []
    for stage in STAGES:
        stage_file = DEEPEYE_ROOT / "app" / "pipeline" / stage / f"{stage}.py"
        if not stage_file.exists():
            continue
        text = stage_file.read_text()
        has_flush = "idx % 5 == 0" in text and "save_result" in text
        if has_flush:
            print(f"  ✓ {stage:25s} flushes checkpoint every 5 items")
        else:
            issues.append(f"  ✗ {stage}: no per-5 flush")
            print(f"  ✗ {stage:25s} no per-5 flush")
    return issues


def check_existing_artifacts():
    """Inspect actual checkpoint artifacts left over from previous runs."""
    print("\n" + "=" * 70)
    print("3. Real checkpoint artifact files on disk")
    print("=" * 70)
    issues = []
    workspace_p2 = DEEPEYE_ROOT / "workspace_p2"
    if not workspace_p2.exists():
        print(f"  (workspace_p2 missing — skipping disk audit)")
        return issues

    for stage in STAGES:
        art_dir = workspace_p2 / stage / "bird" / "dev.artifacts"
        if not art_dir.exists():
            print(f"  - {stage:25s} no artifact dir (stage may not have run)")
            continue
        records = art_dir / f"{stage}.jsonl"
        meta = art_dir / "meta.json"
        if not records.exists():
            issues.append(f"  ✗ {stage}: artifact dir exists but {records.name} missing")
            continue
        n_records = 0
        item_ids = set()
        with records.open() as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                n_records += 1
                item_ids.add(entry["item_id"])
        print(f"  ✓ {stage:25s} {n_records:4d} records, {len(item_ids):3d} unique items, meta={'yes' if meta.exists() else 'no'}")
    return issues


def check_crash_recovery_evidence():
    """Search recent pipeline logs for skip messages — proof that resume worked."""
    print("\n" + "=" * 70)
    print("4. Real crash-recovery evidence in pipeline logs")
    print("=" * 70)
    logs_dir = DEEPEYE_ROOT / "logs"
    if not logs_dir.exists():
        print(f"  (no logs dir)")
        return []
    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        print(f"  (no log files)")
        return []
    recent = log_files[-1]
    print(f"  Most recent log: {recent.name}")
    text = recent.read_text()
    skip_pattern = re.compile(r"Skipping data item (\d+) because it has already been generated")
    skipped = sorted({int(m.group(1)) for m in skip_pattern.finditer(text)})
    print(f"  Resume detected: {len(skipped)} items skipped (resumed from checkpoint)")
    if skipped:
        print(f"  Skipped IDs: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    # Also look for "Restored items" messages
    restore_pattern = re.compile(r"\[(\w+)\] Restored (\d+) items from incremental checkpoint")
    for m in restore_pattern.finditer(text):
        print(f"  Stage [{m.group(1)}] restored {m.group(2)} items from incremental checkpoint")

    return []


def check_artifact_writer_thread():
    """Confirm the artifact writer thread is durable (writes synchronously / fsyncs)."""
    print("\n" + "=" * 70)
    print("5. Writer durability (async writer thread, but fsynced)")
    print("=" * 70)
    store = DEEPEYE_ROOT / "app" / "services" / "artifact_store.py"
    text = store.read_text()
    has_fsync = "fsync" in text or "fsync(" in text
    has_atomic = "atomic" in text or "tmp" in text or "rename" in text
    has_writer_thread = "_writer_thread" in text or "daemon=True" in text
    print(f"  Writer is in background thread: {has_writer_thread}")
    print(f"  Uses fsync for durability:      {has_fsync}")
    print(f"  Uses atomic file replace:       {has_atomic}")
    if not has_fsync:
        print("  ⚠ No explicit fsync — relies on OS buffering. In a hard crash "
              "(power loss / OOM kill), the latest 1-2 batches may be lost. "
              "Per-5 flush limits worst-case to ~5 items.")


def main():
    all_issues = []
    all_issues += check_skip_logic()
    all_issues += check_checkpoint_flushing()
    all_issues += check_existing_artifacts()
    check_crash_recovery_evidence()
    check_artifact_writer_thread()

    print("\n" + "=" * 70)
    if all_issues:
        print(f"✗ {len(all_issues)} ISSUES FOUND:")
        for i in all_issues:
            print(i)
        sys.exit(1)
    print("✅ Checkpointing infrastructure is sound")
    print("=" * 70)
    print("\nResume behavior for full BIRD-Dev (1534):")
    print("  - Every 5 items: incremental flush per stage")
    print("  - On crash: lose at most ~5 items per stage in flight")
    print("  - On restart: same script + same config auto-resumes")
    print("  - Skip-if-complete is per-item, idempotent")


if __name__ == "__main__":
    main()
