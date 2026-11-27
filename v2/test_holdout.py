"""V2 holdout test: one season, every model scored on the same matches with odds.

  python v2/test_holdout.py --repo . --matches <store.csv> --log <player_state_log.csv> --test-year 2026 \
         --out-dir data/v2/local/test_2026

What it does
  1. Pre-match features for every tour-level match from 2022 on, from post-tournament profiles only
     (a tournament sees only snapshots of tournaments that ended before it started).
  2. Trains V2 (and a plain Elo + ranking baseline) on every G/M/A/F match from 2022 up to the year
     before the test year. Same features and settings as the 2025 first look.
  3. Joins the test year's CourtIQ prediction files (odds, V1 RF, V1 CCK). Logged events whose store
     tourney_id is the slug join exactly on slug/round/match_no; older events fall back to player pair
     within 10 days. The result used for scoring is the store's (the actual result), never the tracker's;
     tracker disagreements are counted and listed.
  4. Scores book, V2, plain baseline, V1 RF and V1 CCK on the identical set of matches: log loss, Brier,
     AUC, accuracy, calibration; each model minus the book with tournament-cluster bootstrap intervals;
     by level and by favourite strength.
  5. Does V2 add anything beyond the odds? Blend logit(p) = a + b*logit(book) + c*logit(V2) with
     tournament-clustered errors, scored out of sample (leave one tournament out, and walk forward).
  6. Betting: V1 Pick Score as served, the same formula with V2 in CCK's slot, and the EV filter
     (back a side when p * decimal odds - 1 > tau). Pick Score ROI is compared with a null that keeps the
     market structure and shuffles the model's disagreement with the market within favourite bands.
Odds are the ones in the prediction files (taken when the round was predicted, not closing odds).
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profiles as pf                                         # noqa: E402
from match_store import load_repo_maps, make_alias            # noqa: E402

COLS = ["mult_elo_diff", "mult_selo_diff", "log_rank_gap", "srv_adj", "ret_adj", "lpm", "prev_match_min",
        "min_7d", "sets_7d", "rest_days", "in_event_matches"]
PLAIN = ["elo_diff", "selo_diff", "log_rank_gap"]
LEVELS = ["G", "M", "A", "F"]
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1 / (1 + np.exp(-z))
def imp(o):                                                                # American odds -> implied prob
    o = np.asarray(o, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def dec(o):                                                                # American odds -> decimal
    o = np.asarray(o, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, 1 + 100 / -o, 1 + o / 100)
ll = lambda y, p: -(y * np.log(np.clip(p, EPS, 1 - EPS)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1 - EPS)))


# ---------------------------------------------------------------- features and model
def features(m, log, first_year):
    evs = m[m.event_start >= f"{first_year}-01-01"].tourney_id.unique()
    F = pf.all_event_features(log, m, evs)
    F = F[~F.walkover.astype(bool)].copy()
    F["year"] = pd.to_datetime(F.event_start).dt.year
    ra = F.rank_a.fillna(2000).clip(1, 2000); rb = F.rank_b.fillna(2000).clip(1, 2000)
    F["log_rank_gap"] = np.log(rb) - np.log(ra)
    pp = F.p_point_model.clip(1e-4, 1 - 1e-4)
    F["lpm"] = np.log(pp / (1 - pp))
    return F


def fit(tr, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    mdl = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(((tr[cols] - mu) / sd).values, tr.y)
    return lambda d: mdl.predict_proba(((d[cols] - mu) / sd).values)[:, 1]


# ---------------------------------------------------------------- prediction files
def load_predictions(repo, year):
    rows = []
    for f in sorted((repo / "reports").glob(f"*{year}_*_predictions_cck_complete.csv")):
        slug, rnd = re.match(r"^([a-z]+\d{4})_([A-Z0-9]+)_predictions_cck_complete\.csv$", f.name).groups()
        c = pd.read_csv(f)
        s_path = repo / "reports" / f"{slug}_{rnd}_predictions_complete.csv"
        s = pd.read_csv(s_path) if s_path.exists() else None
        for _, x in c.iterrows():
            p_std = np.nan
            if s is not None:
                hit = s[(s.match_no == x.match_no) & (s.player_a == x.player_a) & (s.player_b == x.player_b)]
                if len(hit) == 1: p_std = float(hit.prob_player_a_win.iloc[0])
            cp = str(x.get("correct_prediction", "")).strip().lower()
            tracker_a = None
            if cp in ("1", "1.0", "true", "0", "0.0", "false"):
                tracker_a = (x.pred_winner == x.player_a) == (cp in ("1", "1.0", "true"))
            rows.append(dict(slug=slug, round=rnd, match_no=int(x.match_no), date=x.get("date"),
                             player_a=x.player_a, player_b=x.player_b,
                             odds_a=pd.to_numeric(x.get("odds_player_a"), errors="coerce"),
                             odds_b=pd.to_numeric(x.get("odds_player_b"), errors="coerce"),
                             p_book_file=pd.to_numeric(x.get("book_fair_prob_a"), errors="coerce"),
                             p_cck=pd.to_numeric(x.get("prob_player_a_win"), errors="coerce"), p_std=p_std,
                             tracker_a=tracker_a))
    return pd.DataFrame(rows)


def join(P, F, m, alias, year):
    S = F[F.year == year].copy()
    S["A"], S["B"] = S.player_a.map(alias), S.player_b.map(alias)
    S["mdate"] = S.match_id.map(m.set_index("match_id").match_date_est)
    P = P.copy(); P["A"], P["B"] = P.player_a.map(alias), P.player_b.map(alias)
    P["mid"] = P.slug + "-" + P["round"] + "-" + P.match_no.astype(str)
    exact_ids = set(S.tourney_id)
    out, how, missing = [], [], []
    sidx = S.set_index("match_id")
    by_pair = {}
    for i, r in S.iterrows(): by_pair.setdefault(frozenset((r.A, r.B)), []).append(i)
    for _, x in P.iterrows():
        if x.slug in exact_ids:
            if x.mid not in sidx.index:
                how.append("exact-missing"); missing.append(f"{x.slug} {x['round']} match_no {x.match_no}: {x.player_a} v {x.player_b}")
                continue
            r = sidx.loc[x.mid].copy(); r["match_id"] = x.mid; kind = "exact"
        else:
            c = [i for i in by_pair.get(frozenset((x.A, x.B)), [])
                 if pd.notna(S.at[i, "mdate"]) and abs((pd.to_datetime(x.date, format="mixed") - S.at[i, "mdate"]).days) <= 10]
            if len(c) > 1:      # same pair twice in 10 days (e.g. ATP Finals round robin, then the final): use the round
                rr = lambda z: "RR" if str(z).startswith("RR") else str(z)
                c = [i for i in c if rr(S.at[i, "round"]) == rr(x["round"])]
            if len(c) != 1: how.append("pair-missing" if not c else "pair-ambiguous"); continue
            r = S.loc[c[0]]; kind = "pair"
        if r.A == x.A and r.B == x.B: flip = False
        elif r.A == x.B and r.B == x.A: flip = True
        else: how.append("names-differ"); continue
        how.append(kind)
        y = 1 - r.y if flip else r.y
        out.append({**x.to_dict(), "join_kind": kind, "y": y, "level": r.level, "surface": r.surface,
                    "match_id": r.match_id,
                    "p_v2": 1 - r.p_v2 if flip else r.p_v2, "p_plain": 1 - r.p_plain if flip else r.p_plain})
    return pd.DataFrame(out), pd.Series(how).value_counts().to_dict(), missing


# ---------------------------------------------------------------- scoring helpers
def cluster_boot(d, groups, n, rng):
    ug = np.unique(groups); idx_by = {g: np.where(groups == g)[0] for g in ug}
    out = np.empty(n)
    for k in range(n):
        s = rng.choice(ug, len(ug)); idx = np.concatenate([idx_by[g] for g in s]); out[k] = d[idx].mean()
    return np.percentile(out, [2.5, 97.5])


def model_table(T, models, rng, n_boot):
    rows = []
    base = ll(T.y.values, T.p_book.values)
    for name, c in models:
        p = T[c].values
        cal = sm.Logit(T.y.values, sm.add_constant(lg(p))).fit(disp=0).params
        d = ll(T.y.values, p) - base
        lo, hi = cluster_boot(d, T.slug.values, n_boot, rng) if c != "p_book" else (0.0, 0.0)
        rows.append(dict(model=name, n=len(T), log_loss=log_loss(T.y, np.clip(p, EPS, 1 - EPS)),
                         brier=brier_score_loss(T.y, p), auc=roc_auc_score(T.y, p), accuracy=((p >= .5) == (T.y == 1)).mean(),
                         cal_slope=cal[1], cal_intercept=cal[0], minus_book_x1000=d.mean() * 1000,
                         ci_lo_x1000=lo * 1000, ci_hi_x1000=hi * 1000))
    return pd.DataFrame(rows)


def blend_tests(T, n_min_prior=200):
    X = pd.DataFrame({"logit_book": lg(T.p_book.values), "logit_v2": lg(T.p_v2.values)})
    groups = pd.factorize(T.slug)[0]
    full = sm.Logit(T.y.values, sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    coef = pd.DataFrame({"coef": full.params, "se_clustered": full.bse, "p": full.pvalues})
    fitlr = lambda Xa, ya: LogisticRegression(C=1e6, tol=1e-10, max_iter=10000).fit(Xa, ya)
    xb, xv = lg(T.p_book.values).reshape(-1, 1), np.column_stack([lg(T.p_book.values), lg(T.p_v2.values)])
    # leave one tournament out
    l0 = l1 = 0.0
    for t in T.slug.unique():
        tr, te = (T.slug != t).values, (T.slug == t).values
        l0 += ll(T.y.values[te], fitlr(xb[tr], T.y.values[tr]).predict_proba(xb[te])[:, 1]).sum()
        l1 += ll(T.y.values[te], fitlr(xv[tr], T.y.values[tr]).predict_proba(xv[te])[:, 1]).sum()
    loto = dict(book_recalibrated=l0 / len(T), book_plus_v2=l1 / len(T), change_x1000=(l1 - l0) / len(T) * 1000)
    # walk forward: each event uses only earlier events of the test year
    order = T.groupby("slug").date.min().sort_values().index.tolist()
    w0 = w1 = 0.0; nw = 0; p_wf = pd.Series(np.nan, index=T.index)
    for i, t in enumerate(order):
        tr = T.slug.isin(order[:i]).values; te = (T.slug == t).values
        if tr.sum() < n_min_prior: continue
        pb = fitlr(xb[tr], T.y.values[tr]).predict_proba(xb[te])[:, 1]
        pv = fitlr(xv[tr], T.y.values[tr]).predict_proba(xv[te])[:, 1]
        w0 += ll(T.y.values[te], pb).sum(); w1 += ll(T.y.values[te], pv).sum(); nw += te.sum()
        p_wf[te] = pv
    wf = dict(n=int(nw), book_recalibrated=w0 / max(nw, 1), book_plus_v2=w1 / max(nw, 1), change_x1000=(w1 - w0) / max(nw, 1) * 1000)
    return coef, loto, wf, p_wf


def pick_score(fa, pA, sA, odds_a, odds_b):
    """V1 Pick Score, vectorised (same formula and thresholds as the engine). pA = model in CCK's slot."""
    ca = pA >= .5; sa = sA >= .5; ba = fa >= .5
    pick_prob = np.where(ca, pA, 1 - pA); other = np.where(ca, sA, 1 - sA)
    bpp = np.where(ca, fa, 1 - fa); odds = np.where(ca, odds_a, odds_b); dog = odds > 0
    edge = pick_prob - bpp
    es = np.clip(edge / .10, 0, 1); cs = np.clip((pick_prob - .5) / .25, 0, 1)
    sag, bag = sa == ca, ba == ca
    ag = np.where(sag & bag, 1.0, np.where(sag, .8, np.where(bag, .5, 0.0)))
    mt = np.clip(1 - np.abs(bpp - .5) / .25, 0, 1); md = np.clip(np.abs(other - pick_prob) / .15, 0, 1)
    ud = np.where(odds < 0, 0, np.where(odds <= 130, .4, np.where(odds <= 180, .7, 1.0)))
    vs = .4 * (1 - cs) + .25 * mt + .2 * md + .15 * ud
    sc = 100 * (.45 * es + .30 * cs + .15 * ag - .10 * vs)
    b = np.full(len(fa), "Avoid", dtype=object)
    lean_small = (edge >= 0) & (edge < .02) & (sc >= 30) & (sc < 45)
    b[lean_small] = "Lean"
    big = edge >= .02
    b[big & (sc >= 20)] = "Upset Watch"; b[big & (sc >= 30)] = "Lean"; b[big & (sc >= 45)] = "Green"; b[big & (sc >= 65)] = "Top Pick"
    o1 = big & dog & (edge >= .06) & (vs >= .65) & (b == "Green"); b[o1] = "Upset Watch"
    o2 = big & (b == "Top Pick") & (vs > .65); b[o2 & dog] = "Upset Watch"; b[o2 & ~dog] = "Green"
    o3 = big & (b == "Green") & (cs < .08); b[o3 & dog] = "Upset Watch"; b[o3 & ~dog] = "Lean"
    return b, ca, odds, bpp, sc


def picks_summary(T, pA, sA, rng, n_null, n_boot, label):
    fa = T.p_book_odds.values
    b, ca, odds, bpp, _ = pick_score(fa, pA, sA, T.odds_a.values, T.odds_b.values)
    won = np.where(ca, T.y.values == 1, T.y.values == 0)
    prof = np.where(won, dec(odds) - 1, -1.0)
    keep = b != "Avoid"
    obs = dict(picks=int(keep.sum()), roi=float(prof[keep].mean()) if keep.any() else np.nan,
               win_minus_fair=float((won[keep] - bpp[keep]).mean()) if keep.any() else np.nan)
    if keep.any():
        lo, hi = cluster_boot(prof[keep], T.slug.values[keep], n_boot, rng); obs.update(roi_ci_lo=lo, roi_ci_hi=hi)
        obs["by_bucket"] = {k: dict(n=int((b == k).sum()), roi=float(prof[b == k].mean())) for k in pd.unique(b[keep])}
    # null: keep each match's market price and the model's structure; shuffle the model's disagreement with
    # the market (from the favourite's side) within favourite-strength bands
    fav = np.where(fa >= .5, 1.0, -1.0)
    band = pd.cut(np.abs(fa - .5), [-.01, .03, .06, .10, .15, .20, .30, .5]).codes
    dA = (lg(np.clip(pA, .01, .99)) - lg(fa)) * fav; dS = (lg(np.clip(sA, .01, .99)) - lg(fa)) * fav
    rois = []
    for _ in range(n_null):
        pa, sa = dA.copy(), dS.copy()
        for g in np.unique(band):
            ix = np.where(band == g)[0]; perm = rng.permutation(len(ix)); pa[ix] = dA[ix][perm]; sa[ix] = dS[ix][perm]
        bb, cc, oo, _, _ = pick_score(fa, sig(lg(fa) + pa * fav), sig(lg(fa) + sa * fav), T.odds_a.values, T.odds_b.values)
        ww = np.where(cc, T.y.values == 1, T.y.values == 0); pr = np.where(ww, dec(oo) - 1, -1.0); kk = bb != "Avoid"
        rois.append(pr[kk].mean() if kk.any() else np.nan)
    rois = np.array(rois)
    obs.update(null_mean_roi=float(np.nanmean(rois)), null_lo=float(np.nanpercentile(rois, 2.5)),
               null_hi=float(np.nanpercentile(rois, 97.5)), p_null_ge_observed=float(np.nanmean(rois >= obs["roi"])))
    return dict(label=label, **obs)


def ev_bets(T, pcol, taus, rng, n_boot):
    ok = T[pcol].notna().values
    D = T[ok]
    da, db = dec(D.odds_a.values), dec(D.odds_b.values)
    ea, eb = D[pcol].values * da - 1, (1 - D[pcol].values) * db - 1
    side = ea >= eb; e = np.where(side, ea, eb); d = np.where(side, da, db)
    won = np.where(side, D.y.values == 1, D.y.values == 0); prof = np.where(won, d - 1, -1.0)
    rows = []
    for tau in taus:
        k = e > tau
        if k.sum() == 0: rows.append(dict(prob=pcol, tau=tau, bets=0)); continue
        lo, hi = cluster_boot(prof[k], D.slug.values[k], n_boot, rng)
        rows.append(dict(prob=pcol, tau=tau, bets=int(k.sum()), win=float(won[k].mean()), roi=float(prof[k].mean()),
                         roi_ci_lo=lo, roi_ci_hi=hi, median_decimal=float(np.median(d[k]))))
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--matches", required=True); ap.add_argument("--log", required=True)
    ap.add_argument("--test-year", type=int, required=True); ap.add_argument("--train-from", type=int, default=2022)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--null-runs", type=int, default=400)
    ap.add_argument("--boot", type=int, default=3000)
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    aliases, _ = load_repo_maps(repo); alias = make_alias(aliases)
    rng = np.random.default_rng(2026)

    m = pf.load_matches(a.matches)
    log = pd.read_csv(a.log)
    F = features(m, log, a.train_from)
    train_years = list(range(a.train_from, a.test_year))
    tr = F[F.year.isin(train_years) & F.level.isin(LEVELS)]
    F["p_v2"], F["p_plain"] = fit(tr, COLS)(F), fit(tr, PLAIN)(F)
    te_all = F[(F.year == a.test_year) & F.level.isin(LEVELS)]
    print(f"store: {len(m)} matches | features from {a.train_from}: {len(F)} | train {train_years[0]}-{train_years[-1]}: {len(tr)} "
          f"| test-year tour-level matches: {len(te_all)}")
    print(f"V2 on all {a.test_year} tour-level matches (with or without odds): log loss {log_loss(te_all.y, te_all.p_v2):.4f}, "
          f"plain {log_loss(te_all.y, te_all.p_plain):.4f}, accuracy {((te_all.p_v2 >= .5) == (te_all.y == 1)).mean():.3f}")

    P = load_predictions(repo, a.test_year)
    J, how, missing = join(P, F, m, alias, a.test_year)
    print(f"\nprediction-file rows: {len(P)} across {P.slug.nunique()} events | join: {how}")
    missing_events = sorted(set(P.slug) - set(J.slug)) if len(J) else sorted(set(P.slug))
    if missing_events: print(f"events with prediction files but no joined matches: {missing_events}")
    if missing:
        print(f"logged rows with no match in the store ({len(missing)}; walkovers are excluded from the store's features, "
              f"and draw changes such as a replaced player land here):")
        for x in missing: print("  " + x)
    trk = J[J.tracker_a.notna()]
    dis = trk[(trk.tracker_a.astype(bool)).astype(float) != trk.y]
    print(f"tracker result disagrees with the store on {len(dis)} joined matches (store result used):")
    for _, r in dis.iterrows(): print(f"  {r.slug} {r['round']} match_no {r.match_no}: {r.player_a} v {r.player_b}")

    # the test set: valid odds and every model present
    J["p_book_odds"] = np.nan
    okodds = J.odds_a.notna() & J.odds_b.notna() & (J.odds_a != 0) & (J.odds_b != 0)
    ia, ib = imp(J.odds_a.where(okodds, 100).values), imp(J.odds_b.where(okodds, 100).values)
    J.loc[okodds, "p_book_odds"] = (ia / (ia + ib))[okodds.values]
    J["p_book"] = J.p_book_file.where(J.p_book_file.notna() & (J.p_book_file > 0) & (J.p_book_file < 1), J.p_book_odds)
    T = J[okodds & J.p_book.notna() & J.p_cck.notna() & J.p_std.notna()].copy().reset_index(drop=True)
    print(f"test set (odds + all models present): {len(T)} matches across {T.slug.nunique()} events | "
          f"levels {T.level.value_counts().to_dict()}")

    models = [("Book (devigged)", "p_book"), ("V2", "p_v2"), ("Plain Elo + ranking", "p_plain"),
              ("V1 RF", "p_std"), ("V1 CCK", "p_cck")]
    R = model_table(T, models, rng, a.boot)
    print("\n== Models on the same matches (minus_book: log loss x1000 vs the book; negative = better than the book) ==")
    print(R.round(4).to_string(index=False))

    T["fav_band"] = pd.cut(np.maximum(T.p_book, 1 - T.p_book), [0, .6, .75, 1], labels=["tight <60%", "moderate 60-75%", "heavy >75%"])
    seg = []
    for key in ("level", "fav_band"):
        for g, x in T.groupby(key, observed=True):
            base = ll(x.y.values, x.p_book.values)
            seg.append(dict(split=key, group=str(g), n=len(x), **{f"{c}_minus_book_x1000": (ll(x.y.values, x[c].values) - base).mean() * 1000
                                                                for c in ("p_v2", "p_std", "p_cck")}))
    S = pd.DataFrame(seg); print("\n== Minus the book by level and favourite strength (x1000) ==\n" + S.round(1).to_string(index=False))

    if T.slug.nunique() < 3:
        print("\n(blend tests need at least 3 events; skipped)")
        coef, loto, wf, p_wf = pd.DataFrame(), {}, {"n": 0}, pd.Series(np.nan, index=T.index)
    else:
        coef, loto, wf, p_wf = blend_tests(T)
    T["p_blend_wf"] = p_wf
    if len(coef):
      print("\n== Does V2 add information beyond the odds? logit(p) = a + b*logit(book) + c*logit(V2) ==")
      print(coef.round(4).to_string())
      print(f"leave one tournament out: book recalibrated {loto['book_recalibrated']:.4f} | book + V2 {loto['book_plus_v2']:.4f} "
          f"| change {loto['change_x1000']:+.2f} x1000")
      print(f"walk forward ({wf['n']} matches after the first {T.slug.nunique() - T[T.p_blend_wf.notna()].slug.nunique()} events): "
          f"book recalibrated {wf['book_recalibrated']:.4f} | book + V2 {wf['book_plus_v2']:.4f} | change {wf['change_x1000']:+.2f} x1000")

    print("\n== Betting (odds from the prediction files) ==")
    PS = [picks_summary(T, T.p_cck.values, T.p_std.values, rng, a.null_runs, a.boot, "V1 Pick Score (CCK model, RF as the other model)"),
          picks_summary(T, T.p_v2.values, T.p_std.values, rng, a.null_runs, a.boot, "Pick Score formula with V2 in CCK's slot")]
    for x in PS:
        print(f"{x['label']}: {x['picks']} non-Avoid picks, ROI {x['roi']:+.2%} "
              f"(95% {x.get('roi_ci_lo', np.nan):+.2%} to {x.get('roi_ci_hi', np.nan):+.2%}), win - fair {x['win_minus_fair']:+.1%}")
        print(f"   null (market kept, model disagreement shuffled): mean ROI {x['null_mean_roi']:+.2%}, 95% {x['null_lo']:+.2%} to "
              f"{x['null_hi']:+.2%}; share of null runs >= observed {x['p_null_ge_observed']:.3f}")
        print(f"   by bucket: " + ", ".join(f"{k} {v['n']} ({v['roi']:+.1%})" for k, v in x.get("by_bucket", {}).items()))
    EV = pd.DataFrame(ev_bets(T, "p_v2", [0, .02, .04, .06, .10], rng, a.boot) + ev_bets(T, "p_blend_wf", [0, .02, .04, .06], rng, a.boot)
                      + ev_bets(T, "p_cck", [0, .02, .04, .06], rng, a.boot))
    print("EV filter (back the side with p * decimal - 1 > tau):\n" + EV.round(4).to_string(index=False))

    T.to_csv(out / f"test_{a.test_year}_matches.csv", index=False)
    summary = dict(test_year=a.test_year, train_years=train_years, join=how, tracker_disagreements=len(dis), n_test=len(T),
                   models=R.to_dict("records"), segments=S.to_dict("records"), blend=coef.reset_index().to_dict("records"),
                   loto=loto, walk_forward=wf, picks=PS, ev=EV.to_dict("records"))
    (out / f"test_{a.test_year}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out / f'test_{a.test_year}_matches.csv'} and the summary JSON")


if __name__ == "__main__":
    main()
