"""Compare two CSVs by value, not by bytes (pandas/numpy versions print numbers and dates differently).

  python v2/compare_csv.py data/v2/matches.csv data/v2/inbox/reference_matches.csv

Same rows, same columns, same blanks; text equal (dates compared as dates, JSON lists element by
element); numbers equal within 1e-6 absolute or 1e-9 relative. Prints the worst difference per column
and exits 1 on any mismatch.
"""
import json, sys
import numpy as np
import pandas as pd

ATOL, RTOL = 1e-6, 1e-9
DATE_HINTS = ("date", "start", "end", "as_of")


def flat(v):
    if isinstance(v, list): return [y for x in v for y in flat(x)]
    return [v]


def json_cells(s):
    out = []
    for v in s:
        if pd.isna(v): out.append(None); continue
        out.append(flat(json.loads(v)))
    return out


a_path, b_path = sys.argv[1], sys.argv[2]
A = pd.read_csv(a_path, dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
B = pd.read_csv(b_path, dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
bad = []
if A.shape != B.shape: bad.append(f"shape {A.shape} vs {B.shape}")
if list(A.columns) != list(B.columns): bad.append(f"columns differ: {sorted(set(A.columns) ^ set(B.columns))}")
print(f"{a_path} vs {b_path}: {len(A)} vs {len(B)} rows")
for c in [c for c in A.columns if c in B.columns] if not bad else []:
    x, y = A[c], B[c]
    if not (x.isna() == y.isna()).all():
        bad.append(f"{c}: blanks in different places ({(x.isna() != y.isna()).sum()} rows)"); continue
    nx, ny = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    if x.notna().sum() and nx.notna().sum() == x.notna().sum() and ny.notna().sum() == y.notna().sum():
        d = (nx - ny).abs().fillna(0); tol = ATOL + RTOL * ny.abs().fillna(0)
        n = int((d > tol).sum()); print(f"  {c:<22} numeric  max diff {d.max():.2e}" + (f"  {n} rows over tolerance" if n else ""))
        if n: bad.append(f"{c}: {n} numeric rows differ, e.g. row {int(d.idxmax())}: {x[d.idxmax()]} vs {y[d.idxmax()]}")
        continue
    if x.dropna().str.startswith("[").all() and x.notna().any():
        jx, jy = json_cells(x), json_cells(y); worst, n = 0.0, 0
        for i, (p, q) in enumerate(zip(jx, jy)):
            if p is None: continue
            if len(p) != len(q): n += 1; continue
            for u, v in zip(p, q):
                if isinstance(u, (int, float)) and isinstance(v, (int, float)):
                    worst = max(worst, abs(u - v)); n += abs(u - v) > ATOL + RTOL * abs(v)
                elif u != v: n += 1
        print(f"  {c:<22} json     max diff {worst:.2e}" + (f"  {n} cells differ" if n else ""))
        if n: bad.append(f"{c}: {n} JSON cells differ")
        continue
    if any(h in c.lower() for h in DATE_HINTS):
        dx, dy = pd.to_datetime(x, errors="coerce", format="mixed"), pd.to_datetime(y, errors="coerce", format="mixed")
        if dx.notna().sum() == x.notna().sum():
            n = int((dx != dy)[x.notna()].sum()); print(f"  {c:<22} date     {n} differ")
            if n: bad.append(f"{c}: {n} dates differ")
            continue
    n = int((x != y)[x.notna()].sum()); print(f"  {c:<22} text     {n} differ")
    if n: bad.append(f"{c}: {n} text values differ, e.g. {x[(x != y) & x.notna()].iloc[0]!r} vs {y[(x != y) & x.notna()].iloc[0]!r}")
print("\nMATCH" if not bad else "\nMISMATCH:\n  " + "\n  ".join(bad))
sys.exit(1 if bad else 0)
