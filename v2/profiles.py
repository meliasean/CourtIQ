"""
CourtIQ V2 post-tournament player profiles, built so predictions can't see the future.

How it works
  1. Read the match store (data/v2/matches.csv).
  2. Replay tournaments in order of their end date. Ratings update match by match inside the replay,
     but nothing is published until the tournament is over.
  3. When a tournament ends, append one row per player who played in it to the state log
     (data/v2/player_state_log.csv), stamped as_of = the tournament's end date. Rows are never
     overwritten or edited: the log is append-only, so every past snapshot can be reproduced.
  4. profile_as_of(date) returns each player's latest row with as_of strictly before that date.
     A tournament starting on date D is predicted from profile_as_of(D), so it only sees
     tournaments that had already finished.

Inside a tournament, only freshness and workload use the tournament's own completed earlier rounds
(minutes on court, matches played this week). Ratings stay frozen at the pre-tournament snapshot.

Usage
  python profiles.py build  [--matches data/v2/matches.csv] [--out data/v2/player_state_log.csv]
  python profiles.py check  [--matches ...] [--log ...]       # leakage tests
  python profiles.py lookup --date 2024-05-26 [--player "Jannik Sinner"]
"""
import argparse, json, sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import courtiq_v2_features as fx          # rating maths shared with the prototype (Rater, p_match, margin_multiplier)

SURFACES = ("Hard", "Clay", "Grass")
WARMUP_END = "2022-01-01"


def _set_warmup_constants(m: pd.DataFrame):
    """Mean winning margin and the points-share -> win mapping, from the warm-up year only."""
    wu = m[(m.event_start < WARMUP_END) & ~m.walkover & ~m.retired]
    fx.MEAN_MARGIN = float(((wu.games_w - wu.games_l) / (wu.games_w + wu.games_l)).mean())
    tpw = ((wu.w_1stWon + wu.w_2ndWon) + (wu.l_svpt - wu.l_1stWon - wu.l_2ndWon)) / (wu.w_svpt + wu.l_svpt)
    tpw = tpw.dropna()
    if len(tpw) > 200:
        from sklearn.linear_model import LogisticRegression
        x = np.concatenate([tpw - 0.5, 0.5 - tpw]).reshape(-1, 1); y = np.r_[np.ones(len(tpw)), np.zeros(len(tpw))]
        fx.TPW_SLOPE = float(LogisticRegression(C=1e6, fit_intercept=False, tol=1e-10, max_iter=10000).fit(x, y).coef_[0][0])


def load_matches(path) -> pd.DataFrame:
    m = pd.read_csv(path, low_memory=False)
    # ISO 8601 accepts both "2026-03-07" and "2025-03-07 02:24:00"; pandas 2 leaves a mixed column as text otherwise
    for c in ("event_start", "event_end", "match_date_est"):
        m[c] = pd.to_datetime(m[c], format="ISO8601")
    for c in ("walkover", "retired"):
        m[c] = m[c].astype(str).str.lower().isin(["true", "1"])
    return m


def _snapshot_rows(R, players, as_of, event):
    rows = []
    for p in players:
        s = R.sr[p]
        recent = [(d.strftime("%Y-%m-%d"), round(mn, 1), st) for d, mn, st, _ in R.hist[p] if (as_of - d).days <= 21]
        last = R.last.get(p)
        rows.append(dict(player=p, as_of=as_of.strftime("%Y-%m-%d"), tourney_id=event,
                         elo=R.elo[p], **{f"selo_{x}": R.selo[(p, x)] for x in SURFACES},
                         melo=R.melo[p], **{f"mselo_{x}": R.mselo[(p, x)] for x in SURFACES},
                         srv=R.srv(p), ret=R.ret(p), **{k: s[k] for k in s},
                         luck_n=R.luck[p][0], luck_w=R.luck[p][1],
                         last_match_date=last[0].strftime("%Y-%m-%d") if last else None,
                         recent_json=json.dumps(recent)))
    return rows


def build(matches: pd.DataFrame) -> pd.DataFrame:
    _set_warmup_constants(matches)
    R = fx.Rater()
    ev = (matches.groupby("tourney_id").agg(start=("event_start", "min"), end=("event_end", "max"))
          .sort_values(["end", "start"]).index)
    rows = []
    for e in ev:
        g = matches[matches.tourney_id == e].sort_values(["round_order", "match_num"], kind="mergesort")
        end = g.event_end.max()
        for r in g.itertuples(index=False):
            if r.walkover: continue
            tpw = ((r.w_1stWon + r.w_2ndWon) + (r.l_svpt - r.l_1stWon - r.l_2ndWon)) / (r.w_svpt + r.l_svpt) \
                if pd.notna(r.w_svpt) and pd.notna(r.l_svpt) and (r.w_svpt + r.l_svpt) > 0 else np.nan
            f_tpw = fx.tpw_to_winprob(tpw) if pd.notna(tpw) else None
            R.update(r.winner, r.loser, r.surface, r.match_date_est, e, r, f_tpw)
        players = pd.unique(g[~g.walkover][["winner", "loser"]].values.ravel())
        rows += _snapshot_rows(R, players, end, e)
        # tour-wide serve baseline, published with the same as_of rule
        rows.append(dict(player="__TOUR__", as_of=end.strftime("%Y-%m-%d"), tourney_id=e,
                         **{f"base_spw_{x}": R.base_spw(x) for x in SURFACES}))
    log = pd.DataFrame(rows)
    log.attrs["mean_margin"], log.attrs["tpw_slope"] = fx.MEAN_MARGIN, fx.TPW_SLOPE
    return log


def profile_as_of(log: pd.DataFrame, date) -> pd.DataFrame:
    """Each player's latest snapshot published strictly before `date`."""
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    x = log[log.as_of < date]
    return x.groupby("player", sort=False).tail(1).set_index("player")


# ---------------- features for one tournament, from its pre-tournament snapshot ----------------
def event_features(log, matches_event: pd.DataFrame, rng=None):
    """Pre-match features for every match of one tournament.
    Ratings come from profile_as_of(event_start); workload adds this tournament's completed earlier rounds."""
    start = matches_event.event_start.min()
    P = profile_as_of(log, start)
    tour = P.loc["__TOUR__"] if "__TOUR__" in P.index else None
    get = lambda p, c, d: (P.at[p, c] if p in P.index and pd.notna(P.at[p, c]) else d)
    rng = rng or np.random.default_rng(7)
    done = {}                                    # player -> list of (date, minutes, sets) played earlier in this event
    out = []
    g = matches_event.sort_values(["round_order", "match_num"], kind="mergesort")
    for ro, rnd in g.groupby("round_order", sort=True):
        for r in rnd.itertuples(index=False):
            a_is_w = rng.random() < 0.5
            A, B = (r.winner, r.loser) if a_is_w else (r.loser, r.winner)
            s = r.surface
            base = tour[f"base_spw_{s}"] if tour is not None else 0.635
            sa, sb = get(A, "srv", 0.0), get(B, "srv", 0.0); ra_, rb_ = get(A, "ret", 0.0), get(B, "ret", 0.0)
            pa, pb = base + sa - rb_, base + sb - ra_
            def work(p):
                hist = [(pd.Timestamp(d), mn, st) for d, mn, st in json.loads(get(p, "recent_json", "[]"))]
                hist += done.get(p, [])
                d0 = r.match_date_est
                last = max([h[0] for h in hist], default=pd.Timestamp(get(p, "last_match_date", "1900-01-01")))
                rest = (d0 - last).days if last.year > 1900 else 60
                w7 = [h for h in hist if 0 <= (d0 - h[0]).days <= 7]; w14 = [h for h in hist if 0 <= (d0 - h[0]).days <= 14]
                ev_prev = done.get(p, [])
                return dict(rest_days=min(max(rest, 0), 60), min_7d=sum(h[1] for h in w7), min_14d=sum(h[1] for h in w14),
                            sets_7d=sum(h[2] for h in w7), prev_match_min=ev_prev[-1][1] if ev_prev else 0.0,
                            in_event_matches=len(ev_prev), layoff=int(rest > 60 and last.year > 1900))
            wa, wb = work(A), work(B)
            la = get(A, "luck_n", 0.0) / (get(A, "luck_w", 0.0) + fx.LUCK_PRIOR)
            lb = get(B, "luck_n", 0.0) / (get(B, "luck_w", 0.0) + fx.LUCK_PRIOR)
            f = dict(match_id=r.match_id, tourney_id=r.tourney_id, event_start=start, level=r.level, surface=s, round=r.round,
                     best_of=r.best_of, player_a=A, player_b=B, y=float(a_is_w), walkover=r.walkover,
                     rank_a=r.winner_rank if a_is_w else r.loser_rank, rank_b=r.loser_rank if a_is_w else r.winner_rank,
                     elo_diff=get(A, "elo", 1500.) - get(B, "elo", 1500.),
                     selo_diff=get(A, f"selo_{s}", 1500.) - get(B, f"selo_{s}", 1500.),
                     mult_elo_diff=get(A, "melo", 1500.) - get(B, "melo", 1500.),
                     mult_selo_diff=get(A, f"mselo_{s}", 1500.) - get(B, f"mselo_{s}", 1500.),
                     srv_adj=sa - sb, ret_adj=ra_ - rb_, p_point_model=fx.p_match(pa, pb, 5 if r.best_of == 5 else 3),
                     luck=la - lb, snapshot_as_of_max=P.as_of.max() if len(P) else None)
            for k in wa: f[k] = wa[k] - wb[k]
            f["layoff_a"], f["layoff_b"] = wa["layoff"], wb["layoff"]
            out.append(f)
        # after the whole round is played, its results feed workload for later rounds (never ratings)
        for r in rnd.itertuples(index=False):
            if r.walkover: continue
            mins = float(r.minutes) if pd.notna(r.minutes) else 0.0
            sets = (r.sets_w + r.sets_l) if pd.notna(r.sets_w) else 0
            for p in (r.winner, r.loser):
                done.setdefault(p, []).append((r.match_date_est, mins, sets))
    return pd.DataFrame(out)


def all_event_features(log, matches, events=None):
    rng = np.random.default_rng(7)
    evs = events if events is not None else matches.tourney_id.unique()
    return pd.concat([event_features(log, matches[matches.tourney_id == e], rng) for e in evs], ignore_index=True)


# ---------------- leakage tests ----------------
def check(matches, log, n_events=6, seed=3):
    ok = True
    rng = np.random.default_rng(seed)
    starts = matches.groupby("tourney_id").event_start.min()
    ends = matches.groupby("tourney_id").event_end.max()
    late = starts[starts >= "2023-01-01"].index.tolist()
    # 1. Every snapshot a tournament can see ended before that tournament started
    bad = 0
    for e in late:
        P = profile_as_of(log, starts[e])
        bad += int((P.as_of >= starts[e].strftime("%Y-%m-%d")).sum())
        own = log[(log.tourney_id == e) & (log.as_of < starts[e].strftime("%Y-%m-%d"))]
        bad += len(own)
    print(f"[1] snapshots visible before their tournament ended: {bad}  {'PASS' if bad == 0 else 'FAIL'}"); ok &= bad == 0
    # 2. Changing a tournament's results (and everything after it) can't change what it was predicted from
    picks = rng.choice(late, n_events, replace=False)
    worst = 0.0
    for e in picks:
        m2 = matches.copy()
        future = m2.event_end >= starts[e]                    # this event and anything not finished before it started
        swap = future & ~m2.walkover
        for a, b in (("winner", "loser"), ("winner_rank", "loser_rank")):
            m2.loc[swap, [a, b]] = m2.loc[swap, [b, a]].values
        for c in fx.STAT_COLS if hasattr(fx, "STAT_COLS") else ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]:
            m2.loc[swap, [f"w_{c}", f"l_{c}"]] = m2.loc[swap, [f"l_{c}", f"w_{c}"]].values
        log2 = build(m2)
        P1, P2 = profile_as_of(log, starts[e]), profile_as_of(log2, starts[e])
        cols = ["elo", "melo", "srv", "ret", "selo_Hard", "mselo_Clay"]
        common = P1.index.intersection(P2.index).drop("__TOUR__", errors="ignore")
        diff = float(np.nanmax(np.abs(P1.loc[common, cols].astype(float).values - P2.loc[common, cols].astype(float).values))) if len(common) else 0.0
        worst = max(worst, diff)
        print(f"    {e}: flipped every result from this tournament on; max change in its pre-tournament profiles = {diff:.2e}")
    print(f"[2] future results changing past profiles: max {worst:.2e}  {'PASS' if worst < 1e-9 else 'FAIL'}"); ok &= worst < 1e-9
    # 3. The log is append-only in time: as_of never decreases
    mono = (pd.to_datetime(log.as_of).diff().dt.days.fillna(0) >= 0).all()
    print(f"[3] state log in time order: {'PASS' if mono else 'FAIL'}"); ok &= bool(mono)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "lookup"])
    ap.add_argument("--matches", default="data/v2/matches.csv")
    ap.add_argument("--out", default="data/v2/player_state_log.csv")
    ap.add_argument("--log", default="data/v2/player_state_log.csv")
    ap.add_argument("--date"); ap.add_argument("--player")
    ap.add_argument("--force", action="store_true", help="allow rewriting past rows (only after a deliberate change)")
    a = ap.parse_args()
    if a.cmd == "build":
        m = load_matches(a.matches); log = build(m)
        out = Path(a.out)
        if out.exists() and not a.force:
            old = pd.read_csv(out)
            new_head = log.iloc[:len(old)].reset_index(drop=True)
            key_ok = len(log) >= len(old) and (old[["player", "as_of", "tourney_id"]].astype(str).values ==
                                               new_head[["player", "as_of", "tourney_id"]].astype(str).values).all()
            num = [c for c in old.columns if c in log.columns and pd.api.types.is_numeric_dtype(old[c])]
            val_ok = key_ok and np.allclose(old[num].values.astype(float), new_head[num].values.astype(float), equal_nan=True, atol=1e-9)
            if not (key_ok and val_ok):
                sys.exit("Refusing to write: the rebuild changes rows already in the published log (a past snapshot would move).\n"
                         "Find out why (edited or reordered past matches, a new alias, changed parameters) before using --force.")
            print(f"Existing {len(old):,} rows unchanged; appending {len(log) - len(old):,} new rows.")
        out.parent.mkdir(parents=True, exist_ok=True); log.to_csv(out, index=False)
        n_ev = log.tourney_id.nunique()
        print(f"{len(log):,} rows ({n_ev} tournaments, {log.player.nunique() - 1:,} players) -> {a.out}")
        print(f"warm-up constants: mean margin {fx.MEAN_MARGIN:.3f}, TPW slope {fx.TPW_SLOPE:.1f}")
    elif a.cmd == "check":
        m = load_matches(a.matches); log = pd.read_csv(a.log)
        sys.exit(0 if check(m, log) else 1)
    else:
        log = pd.read_csv(a.log); P = profile_as_of(log, a.date)
        cols = ["as_of", "tourney_id", "elo", "melo", "selo_Hard", "selo_Clay", "selo_Grass", "srv", "ret", "last_match_date"]
        print(P.loc[[a.player]][cols].T.to_string() if a.player else P[cols].sort_values("melo", ascending=False).head(20).to_string())
