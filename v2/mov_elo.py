"""
Margin-of-victory multiplier Elo for CourtIQ V2: draft + test.

Update rule (winner always gains, loser always loses; the score only scales the size):
    delta = K * M(margin, elo gap) * (1 - E_w)
    E_w   = 1 / (1 + 10^((R_l - R_w)/400))
Margin from the score string: m = (games_w - games_l) / (games_w + games_l), in [0, 1].
Retirements use M = 1 (incomplete score), walkovers don't update.

Variants tested (parameters picked on 2022 only, tested on 2023-24):
    plain    : M = 1
    linear   : M = clip(1 + beta * (m - m_bar) / m_sd, 0.25, 3)
    power    : M = clip((m / m_bar) ** gamma, 0.25, 3)
    sets     : M = 1 + beta * (straight_sets - p_straight)   (score-light version)
    damped   : linear * 2.2 / (2.2 + 0.001 * (R_w - R_l))     (538-style: big wins by big favourites count less)
"""
import itertools, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
sys.path.insert(0, ".")
exec(open("courtiq_v2_features.py").read().split("class Rater")[0])   # ROUND_RANK, games_from_score, etc.

M = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../atp_matches_2021_2024.csv", low_memory=False)
M["tdate"] = pd.to_datetime(M.tourney_date.astype(str), format="%Y%m%d")
M["rr"] = M["round"].map(ROUND_RANK).fillna(3)
M = M.sort_values(["tdate", "tourney_id", "rr", "match_num"], kind="mergesort").reset_index(drop=True)
M["surf"] = M.surface.fillna("Hard").str.title()
g = M.score.map(games_from_score)
M["gw"] = [x[0] if x else np.nan for x in g]; M["gl"] = [x[1] if x else np.nan for x in g]
M["margin"] = (M.gw - M.gl) / (M.gw + M.gl)
M["ret"] = M.score.astype(str).str.contains("RET|DEF|ABD", regex=True)
M["wo"] = M.score.astype(str).str.contains("W/O")
nsets = M.score.astype(str).map(lambda s: len([t for t in s.split() if "-" in t]))
M["straight"] = np.where(M.best_of == 5, nsets == 3, nsets == 2).astype(float)

wu = M[(M.tdate < "2022-01-01") & ~M.ret & ~M.wo & M.margin.notna()]
M_BAR, M_SD, P_STRAIGHT = wu.margin.mean(), wu.margin.std(), wu.straight.mean()
print(f"warm-up 2021: mean margin {M_BAR:.3f}, sd {M_SD:.3f}, straight-sets share {P_STRAIGHT:.3f}")

def multiplier(kind, p, m, straight, ret, gap_w):
    if ret or np.isnan(m) or kind == "plain": base = 1.0
    elif kind in ("linear", "damped"): base = np.clip(1 + p * (m - M_BAR) / M_SD, 0.25, 3)
    elif kind == "power": base = np.clip((max(m, 1e-3) / M_BAR) ** p, 0.25, 3)
    elif kind == "sets": base = 1 + p * (straight - P_STRAIGHT)
    if kind == "damped": base *= 2.2 / (2.2 + 0.001 * gap_w)
    return base

def run(kind, K, p):
    elo, selo = {}, {}
    pre = np.zeros((len(M), 2))
    Ks = 0.75 * K
    for i, r in enumerate(M.itertuples(index=False)):
        w, l, s = r.winner_name, r.loser_name, r.surf
        ew, el = elo.get(w, 1500.), elo.get(l, 1500.); sw, sl = selo.get((w, s), 1500.), selo.get((l, s), 1500.)
        pre[i] = (ew - el, sw - sl)            # winner minus loser, before the match
        if r.wo: continue
        mult = multiplier(kind, p, r.margin, r.straight, r.ret, ew - el)
        e = 1 / (1 + 10 ** ((el - ew) / 400)); d = K * mult * (1 - e)
        elo[w], elo[l] = ew + d, el - d
        e = 1 / (1 + 10 ** ((sl - sw) / 400)); d = Ks * mult * (1 - e)
        selo[(w, s)], selo[(l, s)] = sw + d, sl - d
    return pre

F = pd.read_csv("feats_2021_2024.csv", parse_dates=["tdate"])
sgn = np.where(F.y == 1, 1.0, -1.0)            # A won -> winner-minus-loser equals A-minus-B
mask_eval = F.level.isin(["G", "M", "A", "F"]) & ~F.is_wo
def add(pre, tag):
    F[f"e_{tag}"] = pre[F.idx.values, 0] * sgn; F[f"s_{tag}"] = pre[F.idx.values, 1] * sgn

def ll_fit(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    m = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(((train[cols] - mu) / sd).values, train.y)
    return m.predict_proba(((test[cols] - mu) / sd).values)[:, 1]

# ---- tuning on 2022 (fit + score on 2022; 3 parameters, so in-sample is fine for ranking) ----
GRID = {"plain": [0], "linear": [0.15, 0.25, 0.35, 0.5, 0.75], "power": [0.2, 0.35, 0.5, 0.75, 1.0], "sets": [0.5, 0.75, 1.0, 1.25], "damped": [0.15, 0.25, 0.35, 0.5, 0.75]}
KS = [24, 32, 48, 64, 80, 96, 128]
tune = []
y22 = F[mask_eval & (F.year == 2022)]
for kind, ps in GRID.items():
    for K, p in itertools.product(KS, ps):
        add(run(kind, K, p), "t")
        x = F.loc[y22.index]
        pr = ll_fit(x, x, ["e_t", "s_t"])
        tune.append(dict(kind=kind, K=K, p=p, ll_2022=log_loss(x.y, pr)))
T = pd.DataFrame(tune)
best = T.loc[T.groupby("kind").ll_2022.idxmin()].sort_values("ll_2022")
print(best.round(5).to_string(index=False))
T.to_csv("mov_tuning.csv", index=False)

# ---- test on 2023-24 with the chosen settings; reference = V1-style plain Elo K=24 ----
choices = {"plain_v1": ("plain", 24, 0)}
for _, r in best.iterrows(): choices[r.kind] = (r.kind, int(r.K), float(r.p))
for tag, (kind, K, p) in choices.items(): add(run(kind, K, p), tag)
ra = F.rank_a.fillna(2000).clip(1, 2000); rb = F.rank_b.fillna(2000).clip(1, 2000)
F["log_rank_gap"] = np.log(rb) - np.log(ra)
F["lpm"] = np.log(F.p_point_model.clip(1e-4, 1 - 1e-4) / (1 - F.p_point_model.clip(1e-4, 1 - 1e-4)))
CORE = ["srv_adj", "ret_adj", "lpm", "prev_match_min", "min_7d", "sets_7d", "rest_days", "in_event_matches"]
folds = [(2023, [2022]), (2024, [2022, 2023])]
E = F[mask_eval]
test = E[E.year.isin([2023, 2024])]
def oos(cols):
    p = pd.Series(index=test.index, dtype=float)
    for ty, trs in folds:
        tr = E[E.year.isin(trs)]; te = E[E.year == ty]
        p.loc[te.index] = ll_fit(tr, te, cols)
    return p
def mll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
rng = np.random.default_rng(4)
gid = test.tourney_id.values; ug = np.unique(gid); gix = [np.where(gid == u)[0] for u in ug]
def ci(d):
    gs = np.array([d[i].sum() for i in gix]); gn = np.array([len(i) for i in gix])
    b = [gs[s].sum() / gn[s].sum() for s in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    return np.percentile(b, 2.5) * 1000, np.percentile(b, 97.5) * 1000
rows = []
for ctx, extra in [("Elo only", []), ("Elo + ranking", ["log_rank_gap"]), ("Elo + ranking + serve/return + fatigue", ["log_rank_gap"] + CORE)]:
    ref = mll(test.y, oos([f"e_plain_v1", f"s_plain_v1"] + extra)).values
    for tag in choices:
        l = mll(test.y, oos([f"e_{tag}", f"s_{tag}"] + extra)).values
        lo, hi = ci(l - ref)
        kind, K, p = choices[tag]
        rows.append(dict(context=ctx, variant=tag, K=K, param=p, logloss=l.mean(), vs_plain_x1000=(l - ref).mean() * 1000, ci_lo=lo, ci_hi=hi))
    # continuous-outcome Elo from the prototype, for reference
    l = mll(test.y, oos(["mov_elo_diff"] + extra)).values
    lo, hi = ci(l - ref)
    rows.append(dict(context=ctx, variant="continuous (prototype)", K=60, param=np.nan, logloss=l.mean(), vs_plain_x1000=(l - ref).mean() * 1000, ci_lo=lo, ci_hi=hi))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False)); R.to_csv("mov_results.csv", index=False)

# Sanity: how big are the multipliers for typical scores under the chosen linear setting?
kind, K, p = choices["linear"]
for sc, gw, gl in [("7-6 7-6", 14, 12), ("6-4 6-4", 12, 8), ("6-3 6-2", 12, 5), ("6-1 6-0", 12, 1), ("4-6 7-6 7-5", 18, 17), ("6-4 3-6 6-3", 15, 13)]:
    m = (gw - gl) / (gw + gl)
    print(f"{sc:>12}: margin {m:.2f} -> multiplier {multiplier('linear', p, m, 0, False, 0):.2f}")
