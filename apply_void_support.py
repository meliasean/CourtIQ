#!/usr/bin/env python3
"""
Teach courtiq_engine.py what a voided pick is.

WHY THIS IS NEEDED
------------------
compute_pick_stats() walks reports/*_predictions_cck*.csv. It never reads
live_picks_log.csv, and the string "voided" appears nowhere in the engine.
So marking a pick voided in the pick log has NO effect on the site's stats
card - the pick sits in the pending count forever.

WHY NOT JUST DELETE THE ROW
---------------------------
compute_pick_stats joins the std and cck files by POSITIONAL INDEX:
    std_pA = float(std_df.iloc[idx].get("prob_player_a_win"))
Removing a row from one file and not the other at the same position
silently mis-pairs every later match in that round with the wrong
std_prob_a. Worse than the bug being fixed.

WHAT THIS DOES
--------------
 A  compute_pick_stats(): skip any row whose `voided` column is truthy,
    so it leaves both the numerator and the pending count.
 B  match card JS: render a "void" badge instead of "pending", so the
    match stays visible with an honest label rather than disappearing.

Idempotent. DRY RUN by default.

Usage:
    python apply_void_support.py
    python apply_void_support.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

SKIP = '''            if ps["bucket"] == "Avoid": continue
            # Voided: match never played (pre-match withdrawal, walkover,
            # abandonment). Leaves BOTH the record and the pending count.
            # Not for mid-match retirements - those produce a real winner.
            if str(m.get("voided", "")).strip().lower() in ("true", "1", "yes", "y"):
                continue'''

BADGE = '''  const isVoid = String(m.voided ?? '').trim().toLowerCase() === 'true'
              || String(m.voided ?? '').trim() === '1';
  const badge = isVoid
    ? '<span class="badge" style="background:var(--bg4);color:var(--txt2)">void</span>'
    : (isPend
    ? '<span class="badge badge-pend">pending</span>'
    : (markChip('M', cpModel) + ' ' + markChip('CCK', cpCck) + ' ' + markChip('BK', cpBook)));'''


def main():
    commit = "--commit" in sys.argv
    path = "courtiq_engine.py"
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    if not os.path.exists(path):
        print(f"ABORT: {path} not found"); sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    if 'm.get("voided"' in raw:
        print("SKIP: void support already present."); return

    applied = []

    # ---- A: compute_pick_stats skip --------------------------------------
    hits = [i for i, L in enumerate(lines)
            if L.strip() == 'if ps["bucket"] == "Avoid": continue']
    if len(hits) != 1:
        print(f"ABORT: expected 1 bucket-skip anchor, found {len(hits)}"); sys.exit(1)
    i = hits[0]
    lines[i:i+1] = SKIP.split("\n")
    applied.append(f"A compute_pick_stats skips voided rows (line {i+1})")

    # ---- B: void badge ----------------------------------------------------
    hits = [j for j, L in enumerate(lines) if L.strip() == "const badge = isPend"]
    if len(hits) != 1:
        print(f"ABORT: expected 1 badge anchor, found {len(hits)}"); sys.exit(1)
    j = hits[0]
    end = None
    for k in range(j, min(j + 6, len(lines))):
        if "markChip('BK', cpBook));" in lines[k]:
            end = k; break
    if end is None:
        print("ABORT: could not find end of badge expression"); sys.exit(1)
    lines[j:end+1] = BADGE.split("\n")
    applied.append(f"B match card renders 'void' badge (line {j+1})")

    out = nl.join(lines)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    for a in applied: print(f"APPLY  {a}")
    print("Syntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")


if __name__ == "__main__":
    main()
