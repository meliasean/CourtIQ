"""Rolling-origin test of the V2 prototype features against an Elo + ranking baseline."""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")

F = pd.read_csv("feats_2021_2024.csv", parse_dates=["tdate"])
F = F[F.level.isin(["G", "M", "A", "F"]) & ~F.is_wo].copy()
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
F["lpm"] = lg(F.p_point_model)
ra = F.rank_a.fillna(2000).clip(1, 2000); rb = F.rank_b.fillna(2000).clip(1, 2000)
F["log_rank_gap"] = np.log(rb) - np.log(ra)                 # positive = A better ranked
FAT = ["rest_days", "min_7d", "min_14d", "sets_7d", "prev_match_min", "in_event_matches", "layoff_a", "layoff_b"]
BASE = ["elo_diff", "selo_diff", "log_rank_gap"]
SETS = {
    "Baseline: Elo + surface Elo + ranking": BASE,
    "+ margin-of-victory Elo": BASE + ["mov_elo_diff"],
    "+ serve/return ratings": BASE + ["srv_adj", "ret_adj"],
    "+ point-model probability": BASE + ["lpm"],
    "+ serve/return + point model": BASE + ["srv_adj", "ret_adj", "lpm"],
    "+ fatigue and freshness": BASE + FAT,
    "+ luck (results vs points won)": BASE + ["luck"],
    "Everything": BASE + ["mov_elo_diff", "srv_adj", "ret_adj", "lpm"] + FAT + ["luck"],
    "Elo alone": ["elo_diff"],
    "Point model alone": ["lpm"],
    "Margin-of-victory Elo alone": ["mov_elo_diff"],
}
folds = [(2023, [2022]), (2024, [2022, 2023])]

def fit_pred(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    m = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(((tr[cols] - mu) / sd).values, tr.y)
    return m.predict_proba(((te[cols] - mu) / sd).values)[:, 1], m

preds = {}
for name, cols in SETS.items():
    p = pd.Series(index=F.index, dtype=float)
    for test_year, train_years in folds:
        tr = F[F.year.isin(train_years)]; te = F[F.year == test_year]
        p.loc[te.index], _ = fit_pred(tr, te, cols)
    preds[name] = p
T = F[F.year.isin([2023, 2024])].copy()
base = preds["Baseline: Elo + surface Elo + ranking"].loc[T.index]
def mll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
lb = mll(T.y, base)
rng = np.random.default_rng(11)
groups = T.tourney_id.values
ug = np.unique(groups)
gidx = {g: np.where(groups == g)[0] for g in ug}
res = []
for name in SETS:
    p = preds[name].loc[T.index]
    l = mll(T.y, p); d = (l - lb).values
    gs = np.array([d[gidx[g]].sum() for g in ug]); gn = np.array([len(gidx[g]) for g in ug])
    bs = []
    for _ in range(2000):
        s = rng.integers(0, len(ug), len(ug)); bs.append(gs[s].sum() / gn[s].sum())
    res.append(dict(model=name, n=len(T), logloss=l.mean(), auc=roc_auc_score(T.y, p), acc=((p >= .5) == (T.y == 1)).mean(),
                    vs_base_x1000=d.mean() * 1000, ci_lo=np.percentile(bs, 2.5) * 1000, ci_hi=np.percentile(bs, 97.5) * 1000))
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
print(R.round(4).to_string()); R.to_csv("results_models.csv", index=False)

# Coefficients on the full set (fit on 2022-2023) with significance
tr = F[F.year.isin([2022, 2023])]
cols = SETS["Everything"]
X = (tr[cols] - tr[cols].mean()) / tr[cols].std().replace(0, 1)
m = sm.Logit(tr.y, sm.add_constant(X)).fit(disp=0)
co = pd.DataFrame(dict(coef_per_sd=m.params, se=m.bse, p=m.pvalues)).drop("const")
print(co.round(4).to_string()); co.to_csv("results_coefs.csv")

# Leakage check: AUC by round should be flat (the V1 surface win-rate leak was not)
rows = []
for r, x in T.groupby("round"):
    if len(x) < 150: continue
    rows.append(dict(round=r, n=len(x), auc_elo=roc_auc_score(x.y, x.elo_diff), auc_srv=roc_auc_score(x.y, x.srv_adj),
                     auc_ret=roc_auc_score(x.y, x.ret_adj), auc_pm=roc_auc_score(x.y, x.p_point_model)))
L = pd.DataFrame(rows); L["o"] = L["round"].map({"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3.5})
L = L.sort_values("o").drop(columns="o"); print(L.round(3).to_string()); L.to_csv("results_leakcheck.csv", index=False)

# Where fatigue bites: previous-round minutes, later rounds only
lt = T[T.in_event_matches.abs() >= 0]
T["prev_gap"] = T.prev_min_a - T.prev_min_b
x = T[(T.prev_min_a > 0) & (T.prev_min_b > 0)].copy()
x["gapband"] = pd.cut(x.prev_gap, [-999, -60, -20, 20, 60, 999], labels=["A rested 60+ min more", "A 20-60 less", "within 20 min", "A 20-60 more", "A played 60+ min more"])
x["base_p"] = base.loc[x.index]
fat = x.groupby("gapband", observed=True).agg(n=("y", "size"), win=("y", "mean"), expected=("base_p", "mean")).assign(gap=lambda z: z.win - z.expected)
print(fat.round(3).to_string()); fat.to_csv("results_fatigue.csv")
# Luck: lucky players vs baseline expectation
T["luck_q"] = pd.qcut(T.luck, 5, labels=["A much unluckier", "A unluckier", "even", "A luckier", "A much luckier"])
T["base_p"] = base
lk = T.groupby("luck_q", observed=True).agg(n=("y", "size"), win=("y", "mean"), expected=("base_p", "mean")).assign(gap=lambda z: z.win - z.expected)
print(lk.round(3).to_string()); lk.to_csv("results_luck.csv")
