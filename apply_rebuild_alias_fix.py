#!/usr/bin/env python3
"""
Fix the ALIASES map in rebuild_profiles_new_elo.py.

THE BUG
-------
The rebuild has its own ALIASES map applied to every name via alias(). Two
entries point at the TYPO instead of the correct spelling:

    "Adolfo Daniel Vallejo": "Diego Vallejo"     <- backwards
    "Otto Virtanen":         "Oscar Virtanen"    <- backwards

So the rebuild renames correct names into wrong ones. That is why:
  - Vallejo has 96 match rows under "Adolfo Daniel Vallejo" but his profile
    is filed under "Diego Vallejo"
  - "Oscar Virtanen" appears in profiles with ZERO match rows anywhere - he
    is manufactured from Otto's 87 real rows by this very map

Neither is a stale profile row. Both are produced fresh on every rebuild.

THE FIX
-------
Reverse those two, and extend the map with the merges confirmed by the audit.
Because alias() runs on the way in, ELO replays through the unified identity
automatically - no separate rename pass over the CSVs is needed for profiles.

NOT INCLUDED - the seven one-off typos (Aleksander Vukic, Guy Gen Ouden,
Jan Engel, Luca/Lorenzo Darderi, Ming Zheng, Miroslav Topo, Nikolai Budov
Kjaer). Each has exactly one real match. Run investigate_phantoms.py first:
if the canonical name is absent from that round it is a misspelling and can
be added here; if present, the row may be field-misaligned and needs a look.

Idempotent. DRY RUN by default.

Usage:
    python apply_rebuild_alias_fix.py
    python apply_rebuild_alias_fix.py --commit
"""
import sys, os, shutil, ast, re
from datetime import datetime

NEW_MAP = '''ALIASES = {
    # --- formatting variants -> the spelling profiles should key on ---
    "Felix Auger-Aliassime":        "Felix Auger Aliassime",
    "Botic Van De Zandschulp":      "Botic van de Zandschulp",
    "Jan Lennard Struff":           "Jan-Lennard Struff",
    "Pierre Hugues Herbert":        "Pierre-Hugues Herbert",
    "Marc Andrea Huesler":          "Marc-Andrea Huesler",
    "Chun Hsin Tseng":              "Chun-Hsin Tseng",
    "Jesper De Jong":               "Jesper de Jong",
    "Alex de Minaur":               "Alex De Minaur",
    "Mackenzie Mcdonald":           "Mackenzie McDonald",
    "James Mccabe":                 "James McCabe",
    "Joao Lucas Reis Da Silva":     "Joao Lucas Reis da Silva",
    "Luca van Assche":              "Luca Van Assche",

    # --- CORRECTED 2026-08-23: these two pointed at the typo, so the
    #     rebuild was renaming correct spellings INTO wrong ones. ---
    "Diego Vallejo":                "Adolfo Daniel Vallejo",
    "Daniel Vallejo":               "Adolfo Daniel Vallejo",
    "Oscar Virtanen":               "Otto Virtanen",

    # --- given-name / nickname variants ---
    "Soon Woo Kwon":                "Soonwoo Kwon",
    "Stan Wawrinka":                "Stanislas Wawrinka",
    "Alexander Molcan":             "Alex Molcan",

    # --- shortened vs full name ---
    "Thiago Tirante":               "Thiago Agustin Tirante",
    "Roman Burruchaga":             "Roman Andres Burruchaga",
    "Tomas Barrios Vera":           "Marcelo Tomas Barrios Vera",
    "Chak Lam Coleman Wong":        "Coleman Wong",

    # --- transliteration used consistently by one feed (15 real matches) ---
    "Alexander Shevchenko":         "Aleksandr Shevchenko",

    # --- name-order flip ---
    "Zhang Zhizhen":                "Zhizhen Zhang",

    # --- confirmed typos (canonical absent from the affected round) ---
    "Dino Dedura":                  "Diego Dedura",
}'''


def main():
    commit = "--commit" in sys.argv
    path = "rebuild_profiles_new_elo.py"
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    if not os.path.exists(path):
        print(f"ABORT: {path} not found"); sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    starts = [i for i, L in enumerate(lines) if L.strip().startswith("ALIASES = {")]
    if len(starts) != 1:
        print(f"ABORT: expected 1 ALIASES map, found {len(starts)}"); sys.exit(1)
    i = starts[0]
    j = next((k for k in range(i, min(i + 80, len(lines))) if lines[k].strip() == "}"), None)
    if j is None:
        print("ABORT: could not find the end of the ALIASES map"); sys.exit(1)

    before = nl.join(lines[i:j+1])
    if '"Diego Vallejo":' in before and '"Adolfo Daniel Vallejo"' in before.split('"Diego Vallejo":')[1][:60]:
        print("SKIP: map already corrected."); return

    lines[i:j+1] = NEW_MAP.split("\n")
    out = nl.join(lines)
    if out == raw:
        print("ABORT: edit produced no change."); sys.exit(1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    print(f"APPLY  ALIASES map replaced (lines {i+1}-{j+1})")
    print("\n--- was ---")
    print(before)
    print("\nSyntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")
    print("\nNEXT:")
    print("  python rebuild_profiles_new_elo.py")
    print("  python backfill_last5.py --commit")
    print("  python courtiq_engine.py site --output docs/index.html")


if __name__ == "__main__":
    main()
