"""
CourtIQ V2 feature prototype: time-faithful pre-match features from Sackmann-format match files.

Every feature for a match is computed only from matches that finished before it
(earlier tournaments, or earlier rounds of the same tournament). Ratings update
after the match is recorded, never before.

Features (all oriented as player A minus player B unless noted):
  elo_diff, selo_diff, blend_elo_diff  - V1-style Elo (K=24 overall, 18 surface) and a 50/50 blend
  mov_elo_diff                         - Elo updated on share of games won instead of win/loss (superseded)
  mult_elo_diff, mult_selo_diff        - margin-multiplier Elo: winner always gains, the score scales the step
                                         M = clip((margin / mean_margin) ** 0.5, 0.25, 3), K=64 overall, 48 surface
  srv_adj, ret_adj                     - opponent-adjusted serve / return points-won skill (above tour avg),
                                         recency-weighted by points and shrunk toward long-run then tour average
  p_point_model                        - match-win probability from a point-by-point model fed with the above
  rest_days, min_7d, min_14d, sets_7d, prev_match_min, in_event_matches, layoff
                                       - workload and freshness
  luck                                 - recent wins minus wins expected from share of total points won

Usage:
  python courtiq_v2_features.py <matches.csv> <out.csv>
"""
import sys
from collections import defaultdict, deque
from functools import lru_cache
import numpy as np
import pandas as pd

ROUND_RANK = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "BR": 6.5, "F": 7, "RR": 3.5, "ER": 1}

# ---------------- parameters (set before looking at test years) ----------------
ELO_BASE, K_ELO, K_SELO = 1500.0, 24.0, 18.0
K_MOV = 60.0                 # game-share outcome has lower variance than win/loss, so a larger K
SRV_DECAY_RECENT = 0.93      # per-match weight decay, half-life ~10 matches
SRV_DECAY_LONG = 0.985       # half-life ~45 matches
PRIOR_LONG_PTS = 300.0       # shrink long-run serve/return toward tour average
PRIOR_RECENT_PTS = 250.0     # shrink recent toward long-run
LUCK_DECAY = 0.90
LUCK_PRIOR = 8.0             # matches of shrinkage toward zero luck
K_MULT, K_MULT_SURF = 64.0, 48.0   # margin-multiplier Elo (tuned on 2022, tested on 2023-24)
MULT_GAMMA = 0.5                   # concave: blowouts count more, with diminishing returns
MEAN_MARGIN = 0.228                # overwritten in build() from the warm-up year


# ---------------- point-model math ----------------
def p_game(p):
    q = 1 - p
    return p**4 * (1 + 4*q + 10*q**2) + 20 * p**3 * q**3 * p**2 / (1 - 2*p*q)

def p_tiebreak(pa, pb):
    """A serves first point; serve alternates after point 1 then every two points. pa/pb = serve-point win prob."""
    @lru_cache(None)
    def f(a, b):
        if a >= 7 and a - b >= 2: return 1.0
        if b >= 7 and b - a >= 2: return 0.0
        if a == b and a >= 6:      # deuce-like: two points, one each serve in a pair
            w2 = pa * (1 - pb); l2 = (1 - pa) * pb
            return w2 / (w2 + l2)
        n = a + b
        a_serves = (n % 4 == 0) or (n % 4 == 3)
        p = pa if a_serves else 1 - pb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)

def p_set(pa, pb, a_serves_first=True):
    ga, gb = p_game(pa), p_game(pb)  # A holds with ga, B holds with gb
    tb = p_tiebreak(pa, pb) if a_serves_first else 1 - p_tiebreak(pb, pa)
    @lru_cache(None)
    def f(a, b):
        if a >= 6 and a - b >= 2: return 1.0
        if b >= 6 and b - a >= 2: return 0.0
        if a == 6 and b == 6: return tb
        n = a + b
        a_serve = (n % 2 == 0) == a_serves_first
        p = ga if a_serve else 1 - gb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)

def p_match(pa, pb, best_of=3):
    pa = float(np.clip(pa, 0.40, 0.92)); pb = float(np.clip(pb, 0.40, 0.92))
    s = 0.5 * (p_set(pa, pb, True) + p_set(pa, pb, False))
    need = 2 if best_of == 3 else 3
    @lru_cache(None)
    def f(a, b):
        if a == need: return 1.0
        if b == need: return 0.0
        return s * f(a + 1, b) + (1 - s) * f(a, b + 1)
    return f(0, 0)


# ---------------- helpers ----------------
def games_from_score(score):
    """Games won by winner and loser from a Sackmann score string. Tiebreak points are ignored."""
    if not isinstance(score, str): return None
    gw = gl = 0
    for tok in score.split():
        if "-" not in tok: continue
        a, b = tok.split("-", 1)
        a = a.split("(")[0]; b = b.split("(")[0]
        if a.isdigit() and b.isdigit():
            gw += int(a); gl += int(b)
    return (gw, gl) if gw + gl > 0 else None

def est_offset_days(round_code, draw_size, best_of):
    """Rough day offset of a round from the tournament start date (Sackmann only gives start dates)."""
    r = ROUND_RANK.get(round_code, 3)
    if round_code == "RR": return 2.0
    first = {128: 1, 96: 1, 64: 2, 56: 2, 48: 3, 32: 3, 28: 3}.get(int(draw_size) if pd.notna(draw_size) else 32, 3)
    step = 2.0 if best_of == 5 else 1.1
    return max(0.0, (r - first) * step)


def margin_multiplier(score):
    """Score -> K multiplier. Margin = (games won - games lost) / total games. Retirements and unparseable scores -> 1."""
    s = str(score)
    if any(t in s for t in ("RET", "DEF", "ABD")): return 1.0
    g = games_from_score(s)
    if not g: return 1.0
    m = (g[0] - g[1]) / (g[0] + g[1])
    return float(np.clip((max(m, 1e-3) / MEAN_MARGIN) ** MULT_GAMMA, 0.25, 3.0))


class Rater:
    def __init__(self):
        self.elo = defaultdict(lambda: ELO_BASE)
        self.selo = defaultdict(lambda: ELO_BASE)
        self.mov = defaultdict(lambda: ELO_BASE)
        self.melo = defaultdict(lambda: ELO_BASE)
        self.mselo = defaultdict(lambda: ELO_BASE)
        # serve/return: weighted sums (num = sum w * pts * residual, den = sum w * pts)
        self.sr = defaultdict(lambda: dict(sNl=0., sDl=0., sNr=0., sDr=0., rNl=0., rDl=0., rNr=0., rDr=0.))
        self.base = defaultdict(lambda: [0.0, 0.0])        # surface -> [serve pts won, serve pts]
        self.hist = defaultdict(deque)                       # player -> deque of (date, minutes, sets, event)
        self.last = {}                                       # player -> (date, event, minutes)
        self.luck = defaultdict(lambda: [0.0, 0.0])          # player -> [sum w*resid, sum w]
        self.event_count = defaultdict(int)                  # (player, event) -> matches so far

    # ---- current estimates ----
    def base_spw(self, surf):
        n, d = self.base[surf]
        return n / d if d > 500 else 0.635
    def srv(self, p):
        s = self.sr[p]
        long_ = s["sNl"] / (s["sDl"] + PRIOR_LONG_PTS)
        return (s["sNr"] + PRIOR_RECENT_PTS * long_) / (s["sDr"] + PRIOR_RECENT_PTS)
    def ret(self, p):
        s = self.sr[p]
        long_ = s["rNl"] / (s["rDl"] + PRIOR_LONG_PTS)
        return (s["rNr"] + PRIOR_RECENT_PTS * long_) / (s["rDr"] + PRIOR_RECENT_PTS)
    def luck_of(self, p):
        n, w = self.luck[p]
        return n / (w + LUCK_PRIOR)

    def workload(self, p, date, event):
        h = self.hist[p]
        while h and (date - h[0][0]).days > 14: h.popleft()
        m7 = sum(x[1] for x in h if (date - x[0]).days <= 7)
        m14 = sum(x[1] for x in h)
        s7 = sum(x[2] for x in h if (date - x[0]).days <= 7)
        last = self.last.get(p)
        rest = (date - last[0]).days if last else 60
        prev = last[2] if last and last[1] == event else 0.0
        return dict(rest_days=min(max(rest, 0), 60), min_7d=m7, min_14d=m14, sets_7d=s7, prev_match_min=prev,
                    in_event_matches=self.event_count[(p, event)], layoff=int(rest > 60 if last else 0))

    # ---- update after a match ----
    def update(self, w, l, surf, date, event, row, f_tpw):
        # Elo
        ew, el = self.elo[w], self.elo[l]; e = 1 / (1 + 10 ** ((el - ew) / 400))
        self.elo[w] = ew + K_ELO * (1 - e); self.elo[l] = el - K_ELO * (1 - e)
        sw, sl = self.selo[(w, surf)], self.selo[(l, surf)]; e = 1 / (1 + 10 ** ((sl - sw) / 400))
        self.selo[(w, surf)] = sw + K_SELO * (1 - e); self.selo[(l, surf)] = sl - K_SELO * (1 - e)
        mult = margin_multiplier(row.score)
        mw, ml = self.melo[w], self.melo[l]; e = 1 / (1 + 10 ** ((ml - mw) / 400))
        self.melo[w] = mw + K_MULT * mult * (1 - e); self.melo[l] = ml - K_MULT * mult * (1 - e)
        mw, ml = self.mselo[(w, surf)], self.mselo[(l, surf)]; e = 1 / (1 + 10 ** ((ml - mw) / 400))
        self.mselo[(w, surf)] = mw + K_MULT_SURF * mult * (1 - e); self.mselo[(l, surf)] = ml - K_MULT_SURF * mult * (1 - e)
        g = games_from_score(row.score)
        if g:
            share = g[0] / (g[0] + g[1])
            mw, ml = self.mov[w], self.mov[l]; e = 1 / (1 + 10 ** ((ml - mw) / 400))
            # expected game share is much closer to 0.5 than expected win prob; map via a softer logistic
            e_share = 0.5 + (e - 0.5) * 0.35
            self.mov[w] = mw + K_MOV * (share - e_share); self.mov[l] = ml - K_MOV * (share - e_share)
        # serve / return (only with stats)
        stats_ok = pd.notna(row.w_svpt) and pd.notna(row.l_svpt) and row.w_svpt > 10 and row.l_svpt > 10
        if stats_ok:
            base = self.base_spw(surf)
            w_spw = (row.w_1stWon + row.w_2ndWon) / row.w_svpt
            l_spw = (row.l_1stWon + row.l_2ndWon) / row.l_svpt
            rw_pre, rl_pre, sw_pre, sl_pre = self.ret(w), self.ret(l), self.srv(w), self.srv(l)
            for p, spw, svpt, opp_ret, rpw, rpt, opp_srv in [
                (w, w_spw, row.w_svpt, rl_pre, 1 - l_spw, row.l_svpt, sl_pre),
                (l, l_spw, row.l_svpt, rw_pre, 1 - w_spw, row.w_svpt, sw_pre)]:
                s = self.sr[p]
                res_s = spw - (base - opp_ret)            # serve above what an average server would get vs this returner
                res_r = rpw - ((1 - base) - opp_srv)      # return above what an average returner would get vs this server
                for key, dec in (("l", SRV_DECAY_LONG), ("r", SRV_DECAY_RECENT)):
                    s["sN" + key] = s["sN" + key] * dec + svpt * res_s; s["sD" + key] = s["sD" + key] * dec + svpt
                    s["rN" + key] = s["rN" + key] * dec + rpt * res_r; s["rD" + key] = s["rD" + key] * dec + rpt
            b = self.base[surf]; b[0] += row.w_1stWon + row.w_2ndWon + row.l_1stWon + row.l_2ndWon; b[1] += row.w_svpt + row.l_svpt
        # luck: actual result minus result expected from share of total points won
        if f_tpw is not None:
            for p, won in ((w, 1.0), (l, 0.0)):
                exp = f_tpw if p == w else 1 - f_tpw
                n, wt = self.luck[p]; self.luck[p] = [n * LUCK_DECAY + (won - exp), wt * LUCK_DECAY + 1]
        # workload
        mins = float(row.minutes) if pd.notna(row.minutes) else 0.0
        sets = len([t for t in str(row.score).split() if "-" in t])
        for p in (w, l):
            self.hist[p].append((date, mins, sets, event)); self.last[p] = (date, event, mins)
            self.event_count[(p, event)] += 1


def tpw_to_winprob(tpw_share):
    """Map share of total points won to a win probability. Fixed logistic (slope set from 2021 warm-up)."""
    return 1 / (1 + np.exp(-(tpw_share - 0.5) * TPW_SLOPE))
TPW_SLOPE = 40.0   # overwritten in build() by a fully converged fit on warm-up data


def build(matches: pd.DataFrame, seed=7, warmup_end="2022-01-01"):
    m = matches.copy()
    m["tdate"] = pd.to_datetime(m.tourney_date.astype(str), format="%Y%m%d")
    m["rr"] = m["round"].map(ROUND_RANK).fillna(3)
    m = m.sort_values(["tdate", "tourney_id", "rr", "match_num"], kind="mergesort").reset_index(drop=True)
    m["surf"] = m.surface.fillna("Hard").str.title()
    m["date"] = m.tdate + pd.to_timedelta([est_offset_days(r, d, b) for r, d, b in zip(m["round"], m.draw_size, m.best_of)], unit="D")
    # total-points-won share per match (winner's), used for luck and to calibrate its win mapping on warm-up only
    tpw = ((m.w_1stWon + m.w_2ndWon) + (m.l_svpt - m.l_1stWon - m.l_2ndWon)) / (m.w_svpt + m.l_svpt)
    m["tpw_w"] = tpw
    global TPW_SLOPE, MEAN_MARGIN
    gm = m.score.map(games_from_score)
    marg = pd.Series([(x[0] - x[1]) / (x[0] + x[1]) if x else np.nan for x in gm], index=m.index)
    clean = ~m.score.astype(str).str.contains("RET|DEF|ABD|W/O", regex=True)
    MEAN_MARGIN = float(marg[(m.tdate < warmup_end) & clean].mean())
    wu = m[(m.tdate < warmup_end) & tpw.notna()]
    if len(wu) > 200:
        # symmetric fit: P(win | share) for both orientations
        from sklearn.linear_model import LogisticRegression
        x = np.concatenate([wu.tpw_w - 0.5, 0.5 - wu.tpw_w]).reshape(-1, 1)
        y = np.concatenate([np.ones(len(wu)), np.zeros(len(wu))])
        TPW_SLOPE = float(LogisticRegression(C=1e6, fit_intercept=False, tol=1e-10, max_iter=10000).fit(x, y).coef_[0][0])
    rng = np.random.default_rng(seed)
    R = Rater()
    rows = []
    for i, r in m.iterrows():
        w, l = r.winner_name, r.loser_name
        a_is_w = rng.random() < 0.5
        A, B = (w, l) if a_is_w else (l, w)
        s, ev, d = r.surf, r.tourney_id, r.date
        base = R.base_spw(s)
        pa = base + R.srv(A) - R.ret(B); pb = base + R.srv(B) - R.ret(A)
        bo = 5 if r.best_of == 5 else 3
        wa, wb = R.workload(A, d, ev), R.workload(B, d, ev)
        feat = dict(idx=i, tdate=r.tdate, year=r.tdate.year, tourney_id=ev, tourney=r.tourney_name, level=r.tourney_level,
                    surface=s, round=r["round"], best_of=bo, player_a=A, player_b=B, y=1.0 if a_is_w else 0.0,
                    score=r.score, is_wo=("W/O" in str(r.score)),
                    elo_diff=R.elo[A] - R.elo[B], selo_diff=R.selo[(A, s)] - R.selo[(B, s)],
                    mov_elo_diff=R.mov[A] - R.mov[B],
                    mult_elo_diff=R.melo[A] - R.melo[B], mult_selo_diff=R.mselo[(A, s)] - R.mselo[(B, s)],
                    srv_adj=R.srv(A) - R.srv(B), ret_adj=R.ret(A) - R.ret(B),
                    pa_serve=pa, pb_serve=pb, p_point_model=p_match(pa, pb, bo),
                    luck=R.luck_of(A) - R.luck_of(B),
                    rank_a=r.winner_rank if a_is_w else r.loser_rank, rank_b=r.loser_rank if a_is_w else r.winner_rank)
        feat["blend_elo_diff"] = 0.5 * feat["elo_diff"] + 0.5 * feat["selo_diff"]
        for k in wa: feat[k] = wa[k] - wb[k]
        feat["layoff_a"], feat["layoff_b"] = wa["layoff"], wb["layoff"]
        feat["prev_min_a"], feat["prev_min_b"] = wa["prev_match_min"], wb["prev_match_min"]
        rows.append(feat)
        f_tpw = tpw_to_winprob(r.tpw_w) if pd.notna(r.tpw_w) else None
        if "W/O" not in str(r.score):
            R.update(w, l, s, d, ev, r, f_tpw)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    df = build(pd.read_csv(src, low_memory=False))
    df.to_csv(out, index=False)
    print(f"{len(df)} matches -> {out}; TPW slope {TPW_SLOPE:.1f}; mean margin {MEAN_MARGIN:.3f}")
