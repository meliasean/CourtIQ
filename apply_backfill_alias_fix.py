#!/usr/bin/env python3
"""
Teach backfill_last5.py the same alias map the rebuild uses.

THE BUG
-------
rebuild_profiles_new_elo.py folds alternate spellings into a canonical name via
its ALIASES map. backfill_last5.py matches derived history by exact raw text and
has no alias awareness, so after the rebuild the two disagree:

  - a player whose prediction rows use ONLY a non-canonical spelling gets no
    history at all, and fillna() leaves the rebuild's naive (non
    boundary-corrected) streak in place
  - WORSE: a player with rows under BOTH spellings gets a PARTIAL history -
    only the canonical-spelled subset - so their streak and last5 are computed
    from a fragment rather than the whole chain

Both were visible as "history but no profile row" for 10 players after the
2026-08-23 alias fix.

THE FIX
-------
Import alias() from rebuild_profiles_new_elo (it is guarded by
if __name__ == "__main__", so importing is safe) and apply it to winner and
loser names as history is built. Importing rather than copying the map means
the two cannot drift apart later.

Idempotent. DRY RUN by default.

Usage:
    python apply_backfill_alias_fix.py
    python apply_backfill_alias_fix.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

IMPORT_OLD = "from courtiq_engine import TOURNEY_ORDER, ROUND_ORDER"
IMPORT_NEW = '''from courtiq_engine import TOURNEY_ORDER, ROUND_ORDER

# Share the rebuild's alias map so profile names and derived history cannot
# disagree. rebuild_profiles_new_elo guards its entry point, so this is safe.
try:
    from rebuild_profiles_new_elo import alias as _alias
except Exception as _e:  # pragma: no cover
    print(f"WARNING: could not import alias() from rebuild_profiles_new_elo ({_e}).")
    print("         Falling back to identity - streaks for merged players will be wrong.")
    def _alias(n):
        return str(n).strip()'''

BODY_OLD = '''            results.append((date, win, "W", k))
            results.append((date, lose, "L", k))'''
BODY_NEW = '''            # Canonicalise before recording, so history for a player split
            # across spellings forms one chain rather than several fragments.
            results.append((date, _alias(win), "W", k))
            results.append((date, _alias(lose), "L", k))'''

DEDUPE_OLD = "            dedupe = (date, a, b)"
DEDUPE_NEW = "            dedupe = (date, _alias(a), _alias(b))"


def main():
    commit = "--commit" in sys.argv
    path = "backfill_last5.py"
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    if not os.path.exists(path):
        print(f"ABORT: {path} not found"); sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"

    if "_alias" in raw:
        print("SKIP: alias awareness already present."); return

    edits = [(IMPORT_OLD, IMPORT_NEW, "import alias() from the rebuild"),
             (BODY_OLD,   BODY_NEW,   "canonicalise winner/loser names"),
             (DEDUPE_OLD, DEDUPE_NEW, "canonicalise the dedupe key")]

    out = raw
    for old, new, label in edits:
        old_n = old.replace("\n", nl)
        if out.count(old_n) != 1:
            print(f"ABORT: expected exactly 1 match for '{label}', "
                  f"found {out.count(old_n)}.")
            print("       Your file differs from the expected shape - send it over.")
            sys.exit(1)
        out = out.replace(old_n, new.replace("\n", nl), 1)
        print(f"APPLY  {label}")

    if out == raw:
        print("ABORT: edit produced no change."); sys.exit(1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)
    print("Syntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")
    print("\nNEXT:")
    print("  python backfill_last5.py            # 'history but no profile row' should shrink")
    print("  python backfill_last5.py --commit")
    print("  python courtiq_engine.py site --output docs/index.html")


if __name__ == "__main__":
    main()
