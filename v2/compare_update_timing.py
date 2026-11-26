"""Post-tournament snapshots vs updating after every match: what does freezing ratings during a tournament cost?"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import profiles as pf

PROTO_FEATS = sys.argv[1] if len(sys.argv) > 1 else "v2/feats_2021_2024.csv"
RAW = sys.argv[2] if len(sys.argv) > 2 else "atp_matches_2021_2024.csv"

m = pf.load_matches("data/v2/matches.csv")
log = pd.read_csv("data/v2/player_state_log.csv")
ev = m[m.event_start >= "2022-01-01"].tourney_id.unique()
post = pf.all_event_features(log, m, ev)
post["year"] = pd.to_datetime(post.event_start).dt.year

# prototype (updated after every match): map its rows to match_id
raw = pd.read_csv(RAW, low_memory=False)
raw["tdate"] = pd.to_datetime(raw.tourney_date.astype(str), format="%Y%m%d")
RR = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "BR": 6.5, "F": 7, "RR": 3.5, "ER": 1}
raw["rr"] = raw["round"].map(RR).fillna(3)
raw = raw.sort_values(["tdate", "tourney_id", "rr", "match_num"], kind="mergesort").reset_index(drop=True)
raw["match_id"] = raw.tourney_id.astype(str) + "-" + raw["round"].astype(str) + "-" + raw.match_num.astype(str)
proto = pd.read_csv(PROTO_FEATS)
proto["match_id"] = raw.match_id.values[proto.idx.values]

keep = lambda d: d[d.level.isin(["G", "M", "A", "F"])]
post = keep(post[~post.walkover.astype(bool)]); proto = keep(proto[~proto.is_wo])
common = set(post.match_id) & set(proto.match_id)
post = post[post.match_id.isin(common)].set_index("match_id"); proto = proto[proto.match_id.isin(common)].set_index("match_id")
proto = proto.loc[post.index]
print(f"matches compared: {len(post):,} (2022 train / 2023-24 test)")

def prep(d):
    d = d.copy()
    ra = d.rank_a.fillna(2000).clip(1, 2000); rb = d.rank_b.fillna(2000).clip(1, 2000)
    d["log_rank_gap"] = np.log(rb) - np.log(ra)
    d["lpm"] = np.log(d.p_point_model.clip(1e-4, 1 - 1e-4) / (1 - d.p_point_model.clip(1e-4, 1 - 1e-4)))
    return d
post, proto = prep(post), prep(proto)
FAT = ["prev_match_min", "min_7d", "sets_7d", "rest_days", "in_event_matches"]
SETS = {"Plain Elo + ranking": ["elo_diff", "selo_diff", "log_rank_gap"],
        "Margin Elo + ranking": ["mult_elo_diff", "mult_selo_diff", "log_rank_gap"],
        "Margin Elo + ranking + serve/return + fatigue": ["mult_elo_diff", "mult_selo_diff", "log_rank_gap", "srv_adj", "ret_adj", "lpm"] + FAT}
folds = [(2023, [2022]), (2024, [2022, 2023])]
def oos(d, cols):
    p = pd.Series(index=d.index, dtype=float)
    for ty, trs in folds:
        tr = d[d.year.isin(trs)]; te = d[d.year == ty]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        mdl = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(((tr[cols] - mu) / sd).values, tr.y)
        p.loc[te.index] = mdl.predict_proba(((te[cols] - mu) / sd).values)[:, 1]
    return p
post["year"] = post.year; proto["year"] = pd.to_datetime(proto.tdate).dt.year
T = post.year.isin([2023, 2024])
def mll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
rng = np.random.default_rng(9)
tid = post.loc[T, "tourney_id"].values; ug = np.unique(tid); gix = [np.where(tid == u)[0] for u in ug]
rows = []
for name, cols in SETS.items():
    lp = mll(post.loc[T, "y"], oos(post, cols)[T]).values
    lq = mll(proto.loc[T, "y"], oos(proto, cols)[T]).values
    d = lp - lq
    gs = np.array([d[i].sum() for i in gix]); gn = np.array([len(i) for i in gix])
    bs = [gs[s].sum() / gn[s].sum() for s in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    rows.append(dict(model=name, n=int(T.sum()), per_match=lq.mean(), post_tournament=lp.mean(),
                     cost_x1000=d.mean() * 1000, ci_lo=np.percentile(bs, 2.5) * 1000, ci_hi=np.percentile(bs, 97.5) * 1000))
R = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False)); R.to_csv(Path(__file__).resolve().parent / "timing_results.csv", index=False)
# where the cost lands: by round, for the full model
lp = mll(post.loc[T, "y"], oos(post, SETS["Margin Elo + ranking + serve/return + fatigue"])[T])
lq = mll(proto.loc[T, "y"], oos(proto, SETS["Margin Elo + ranking + serve/return + fatigue"])[T])
by = pd.DataFrame({"round": post.loc[T, "round"], "d": (lp - lq).values * 1000}).groupby("round").d.agg(["size", "mean"]).round(2)
print(by.reindex([r for r in ["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR"] if r in by.index]).to_string())
