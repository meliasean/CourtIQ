#!/usr/bin/env python3
"""
Locate every alias variant across CourtIQ files. READ ONLY - writes nothing
except an optional report CSV.

For each variant pair it reports:
  - which files and columns hold it, and how many rows
  - the DATE RANGE each variant covers
  - whether the two ranges OVERLAP

The date-overlap test is the important one. A feed that renamed a player
mid-season produces two DISJOINT ranges (variant A stops, variant B starts).
Two genuinely different people produce OVERLAPPING ranges. Overlap is not
proof of anything, but it means look closer before merging.

Also flags any match where two variants of the same player face each other -
that would mean the mapping is wrong.

Usage:
    python locate_player_aliases.py
    python locate_player_aliases.py --dir reports --out alias_report.csv
"""
import sys, os, glob
from collections import defaultdict
import pandas as pd

# canonical -> list of every spelling seen for that player
GROUPS = {
    "Adolfo Daniel Vallejo": ["Adolfo Daniel Vallejo", "Diego Vallejo", "Daniel Vallejo", "Adolfo Vallejo", "Dani Vallejo"],
    "Felix Auger-Aliassime": ["Felix Auger-Aliassime", "Felix Auger Aliassime"],
    "Pierre-Hugues Herbert": ["Pierre-Hugues Herbert", "Pierre Hugues Herbert"],
    "Marc-Andrea Huesler":   ["Marc-Andrea Huesler", "Marc Andrea Huesler"],
    "Jan-Lennard Struff":    ["Jan-Lennard Struff", "Jan Lennard Struff"],
    "Chun-Hsin Tseng":       ["Chun-Hsin Tseng", "Chun Hsin Tseng"],
    "Jesper de Jong":        ["Jesper de Jong", "Jesper De Jong"],
    "Alex de Minaur":        ["Alex de Minaur", "Alex De Minaur"],
    "Botic van de Zandschulp": ["Botic van de Zandschulp", "Botic Van De Zandschulp"],
    "Mackenzie McDonald":    ["Mackenzie McDonald", "Mackenzie Mcdonald"],
    "James McCabe":          ["James McCabe", "James Mccabe"],
    "Joao Lucas Reis da Silva": ["Joao Lucas Reis da Silva", "Joao Lucas Reis Da Silva"],
    "Luca Van Assche":       ["Luca Van Assche", "Luca van Assche"],
    "Soonwoo Kwon":          ["Soonwoo Kwon", "Soon Woo Kwon"],
    "Stanislas Wawrinka":    ["Stanislas Wawrinka", "Stan Wawrinka"],
    "Alex Molcan":           ["Alex Molcan", "Alexander Molcan"],
    "Thiago Agustin Tirante":["Thiago Agustin Tirante", "Thiago Tirante"],
    "Roman Andres Burruchaga":["Roman Andres Burruchaga", "Roman Burruchaga"],
    "Marcelo Tomas Barrios Vera":["Marcelo Tomas Barrios Vera", "Tomas Barrios Vera"],
    "Coleman Wong":          ["Coleman Wong", "Chak Lam Coleman Wong"],
    "Nicolai Budkov Kjaer":  ["Nicolai Budkov Kjaer", "Nikolai Budov Kjaer"],
    "Aleksandr Shevchenko":  ["Aleksandr Shevchenko", "Alexander Shevchenko"],
    "Aleksandar Vukic":      ["Aleksandar Vukic", "Aleksander Vukic"],
    "Guy Den Ouden":         ["Guy Den Ouden", "Guy Gen Ouden"],
    "Diego Dedura":          ["Diego Dedura", "Dino Dedura"],
    "Luciano Darderi":       ["Luciano Darderi", "Luca Darderi", "Lorenzo Darderi"],
    "Zhizhen Zhang":         ["Zhizhen Zhang", "Zhang Zhizhen", "Jiri Zhang"],
    # unconfirmed - listed so we can SEE where they live before deciding
    "?? Marko Topo":         ["Marko Topo", "Miroslav Topo"],
    "?? Michael Zheng":      ["Michael Zheng", "Ming Zheng"],
    "?? Luis Felipe Miguel": ["Luis Felipe Miguel", "Luis Miguel"],
    "?? Otto Virtanen":      ["Otto Virtanen", "Oscar Virtanen"],
    "?? Justin Engel":       ["Justin Engel", "Jan Engel"],
}

NAME_COLS = ["player_a", "player_b", "pred_winner", "actual_winner",
             "winner", "loser", "player", "player_name", "name",
             "opponent", "book_pick"]
DATE_COLS = ["date", "match_date", "played_on", "tourney_date"]


def kind(path):
    b = os.path.basename(path).lower()
    if "profile" in b:            return "PROFILE"
    if "picks_log" in b or "live_picks" in b: return "PICKLOG"
    if "predictions" in b:        return "PREDICTION"
    if "draw" in b:               return "DRAW"
    return "OTHER"


def main():
    target = "reports"
    if "--dir" in sys.argv:
        target = sys.argv[sys.argv.index("--dir") + 1]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    roots = [target, "live_log", "."]
    files = []
    for r in roots:
        files += glob.glob(os.path.join(r, "*.csv"))
    files = sorted(set(files))

    variant_to_group = {}
    for canon, variants in GROUPS.items():
        for v in variants:
            variant_to_group[v] = canon

    # variant -> {file: {col: count}}, variant -> [dates], variant -> set(kinds)
    where = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    dates = defaultdict(list)
    kinds = defaultdict(set)
    conflicts = []
    rows_out = []

    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        cols = [c for c in NAME_COLS if c in df.columns]
        if not cols:
            continue
        k = kind(path)
        dcol = next((c for c in DATE_COLS if c in df.columns), None)

        for c in cols:
            s = df[c].astype(str).str.strip()
            for variant in variant_to_group:
                mask = s == variant
                n = int(mask.sum())
                if not n:
                    continue
                where[variant][path][c] = n
                kinds[variant].add(k)
                if dcol is not None:
                    dates[variant] += [d for d in df.loc[mask, dcol].dropna().astype(str)]

        # same-player-vs-self check
        if "player_a" in df.columns and "player_b" in df.columns:
            a = df["player_a"].astype(str).str.strip().map(variant_to_group)
            b = df["player_b"].astype(str).str.strip().map(variant_to_group)
            bad = df[(a.notna()) & (a == b)]
            for _, r in bad.iterrows():
                conflicts.append((path, r["player_a"], r["player_b"]))

    print("=" * 72)
    print(f"ALIAS LOCATION REPORT  -  {len(files)} CSV(s) scanned")
    print("=" * 72)

    for canon, variants in GROUPS.items():
        present = [v for v in variants if v in where]
        if len(present) < 2:
            if present:
                print(f"\n{canon}\n  only one spelling present: {present[0]!r} - no split")
            else:
                print(f"\n{canon}\n  not found")
            continue

        print(f"\n{canon}")
        spans = {}
        for v in present:
            total = sum(sum(cc.values()) for cc in where[v].values())
            ds = sorted(d for d in dates[v] if d and d.lower() != "nan")
            span = (ds[0], ds[-1]) if ds else None
            spans[v] = span
            span_s = f"{span[0]} .. {span[1]}" if span else "no dates"
            print(f"    {v!r}  -  {total} row(s)   [{','.join(sorted(kinds[v]))}]   {span_s}")
            for path, cc in sorted(where[v].items()):
                detail = ", ".join(f"{c}={n}" for c, n in sorted(cc.items()))
                print(f"         {os.path.relpath(path)}  ({detail})")
                rows_out.append({"canonical": canon, "variant": v,
                                 "file": os.path.relpath(path), "detail": detail,
                                 "kind": kind(path),
                                 "first_date": span[0] if span else "",
                                 "last_date": span[1] if span else ""})

        dated = {v: s for v, s in spans.items() if s}
        if len(dated) >= 2:
            vs = list(dated)
            overlap = False
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a1, a2 = dated[vs[i]]
                    b1, b2 = dated[vs[j]]
                    if a1 <= b2 and b1 <= a2:
                        overlap = True
                        print(f"    ** OVERLAP: {vs[i]!r} and {vs[j]!r} share a date range")
            if not overlap:
                print(f"    -> ranges are disjoint: consistent with a feed rename")

    print("\n" + "=" * 72)
    if conflicts:
        print(f"CONFLICTS - {len(conflicts)} match(es) where two variants of the SAME")
        print("player face each other. At least one grouping above is wrong:")
        for path, a, b in conflicts:
            print(f"    {os.path.relpath(path)}: {a!r} vs {b!r}")
    else:
        print("No same-player-vs-self matches. All groupings are internally consistent.")

    if out and rows_out:
        pd.DataFrame(rows_out).to_csv(out, index=False)
        print(f"\nWrote {out} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
