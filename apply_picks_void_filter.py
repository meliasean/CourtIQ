#!/usr/bin/env python3
"""
Exclude voided picks from the picks page.

buildPicks() filters only on bucket:
    const picks=r.matches.filter(m=>m.bucket&&m.bucket!=='Avoid');
so a voided match still renders a card. compute_pick_stats() already skips
voided rows, which left the page and the stats card disagreeing.

Because total++ runs inside the loop over `picks`, filtering that one array
also corrects the per-round "N picks" label and the "(N active)" header.

The truthiness check is deliberately string-based: the voided value arrives
as JSON true, the string "True", 1, or null/NaN depending on how pandas
serialized the column, and String(x).toLowerCase() handles all of them.

Idempotent. DRY RUN by default.

Usage:
    python apply_picks_void_filter.py
    python apply_picks_void_filter.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

OLD = "const picks=r.matches.filter(m=>m.bucket&&m.bucket!=='Avoid');"
NEW = ("const picks=r.matches.filter(m=>m.bucket&&m.bucket!=='Avoid'"
       "&&!['true','1','yes'].includes(String(m.voided??'').trim().toLowerCase()));")


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

    if "String(m.voided" in raw and "r.matches.filter" in raw:
        hits = [L for L in lines if "r.matches.filter" in L and "m.voided" in L]
        if hits:
            print("SKIP: picks page already filters voided."); return

    hits = [i for i, L in enumerate(lines) if L.strip() == OLD]
    if len(hits) != 1:
        print(f"ABORT: expected 1 picks filter, found {len(hits)}")
        for i, L in enumerate(lines):
            if "r.matches.filter" in L:
                print(f"    line {i+1}: {L.strip()[:110]}")
        sys.exit(1)

    i = hits[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = indent + NEW

    out = nl.join(lines)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    print(f"APPLY  buildPicks() excludes voided (line {i+1})")
    print("       also corrects the per-round count and the (N active) header,")
    print("       since total++ runs inside the loop over this array")
    print("Syntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")
    print("\nNEXT:  python courtiq_engine.py site --output docs/index.html")


if __name__ == "__main__":
    main()
