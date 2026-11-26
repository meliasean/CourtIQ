"""
CourtIQ V2 match store: one canonical, append-only table of completed matches with scores.

Inputs (any mix, later files never overwrite earlier matches):
  - Sackmann-format yearly files (atp_matches_YYYY.csv), e.g. 2021-2025
  - CourtIQ results files for 2026+ in the minimal template (see TEMPLATE_COLUMNS)

Output: data/v2/matches.csv, plus a validation report printed to the console.

Each match carries the dates the profile builder needs to stay time-faithful:
  event_start  - first day of the tournament
  event_end    - day of the final (start + a duration by level and draw size, checked against the 2024 calendar)
A tournament's results only become visible to tournaments that start after its event_end.

Usage:
  python match_store.py --sackmann data/v2/raw/atp_matches_2021.csv ... --courtiq data/v2/raw/results_2026.csv --out data/v2/matches.csv
"""
import argparse, re, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROUND_ORDER = {"Q1": 0, "Q2": 0, "Q3": 0, "ER": 1, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 4, "RR1": 4, "RR2": 4, "RR3": 4,
               "QF": 5, "SF": 6, "BR": 7, "F": 7}
STAT_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
TEMPLATE_COLUMNS = ["tourney_slug", "event_start", "round", "match_no", "winner", "loser", "score",   # required
                    "minutes", "match_date", "event_end", "winner_rank", "loser_rank"]               # optional
# Serve stats are optional in the template too: w_svpt, w_1stIn, ... l_bpFaced (Sackmann names).


def _dict_literal(src: str, name: str) -> dict:
    start = src.index(f"{name}") ; start = src.index("{", start)
    depth, i = 0, start
    while True:
        if src[i] == "{": depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0: break
        i += 1
    ns = {}; exec(f"x = {src[start:i + 1]}", ns); return ns["x"]

def load_repo_maps(repo: Path):
    eng = (repo / "courtiq_engine.py").read_text(encoding="utf-8")
    aliases = dict(_dict_literal(eng, "ALIASES: Dict[str, str] ="))
    reb = repo / "rebuild_profiles_new_elo.py"
    if reb.exists():
        aliases.update(_dict_literal(reb.read_text(encoding="utf-8"), "ALIASES = "))
    configs = _dict_literal(eng, "TOURNAMENT_CONFIGS: Dict[str, dict] =")
    v2 = Path(__file__).with_name("v2_name_aliases.csv")
    if v2.exists():   # V2 alias table: confirmed rows only; 'review' rows are left unmapped until checked
        t = pd.read_csv(v2).fillna("")
        for _, r in t[t.status == "confirmed"].iterrows():
            aliases[r.variant] = r.canonical
            for k, v in list(aliases.items()):        # keep chains pointing at the final canonical name
                if v == r.variant: aliases[k] = r.canonical
    return aliases, configs

def make_alias(aliases):
    def alias(n):
        if n is None or (isinstance(n, float) and np.isnan(n)): return n
        n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip()
        n = re.sub(r"\s+", " ", n)
        return aliases.get(n, n)
    return alias


def parse_score(score):
    """-> games_w, games_l, sets_w, sets_l, retired, walkover. Tiebreak points ignored; a 7-6 set counts 7-6."""
    s = str(score) if isinstance(score, str) else ""
    wo = "W/O" in s.upper() or "WALKOVER" in s.upper()
    ret = bool(re.search(r"\b(RET|DEF|ABD|ABN)\b", s.upper()))
    gw = gl = sw = sl = 0
    for tok in s.split():
        mt = re.match(r"^\[(\d+)-(\d+)\]$", tok)          # match tiebreak in place of a final set: counts as a 1-0 set
        if mt:
            a, b = int(mt.group(1)), int(mt.group(2))
            gw += int(a > b); gl += int(b > a); sw += int(a > b); sl += int(b > a); continue
        m = re.match(r"^(\d+)-(\d+)", tok)
        if not m: continue
        a, b = int(m.group(1)), int(m.group(2))
        gw += a; gl += b
        if a > b: sw += 1
        elif b > a: sl += 1
    if gw + gl == 0: gw = gl = sw = sl = np.nan
    return gw, gl, sw, sl, ret, wo


def event_duration_days(level, draw_size, best_of):
    """Days from event_start (Sackmann's Monday date) to the final, checked against the 2024 calendar:
    Slams and 96/128-draw Masters finish on day +13, 56/64-draw Masters and 250/500s on day +6
    (Canada/Cincinnati Monday finals land on +7, the same day the next event starts, which only
    matters for the two finalists, whose next match is later in the week anyway)."""
    d = int(draw_size) if pd.notna(draw_size) else 32
    if level == "G": return 13
    if level == "M": return 13 if d >= 96 else 6
    if level == "F": return 6
    if level in ("D", "O"): return 10        # Davis Cup Finals, United Cup, Olympics, Laver Cup: kept wide
    return 6                                  # ATP 250/500, Monday to Sunday


def est_match_offset(round_code, draw_size, level):
    r = ROUND_ORDER.get(str(round_code), 3)
    first = {128: 1, 96: 1, 64: 2, 56: 2, 48: 3, 32: 3, 28: 3}.get(int(draw_size) if pd.notna(draw_size) else 32, 3)
    step = 2.0 if level == "G" else 1.1
    if str(round_code).startswith("RR"): return 2.0
    return max(0.0, (r - first) * step)


def _ymd(col):
    """YYYYMMDD column -> dates, whether pandas read it as text, int or float (blanks make it float in pandas 2)."""
    v = pd.to_numeric(col, errors="coerce")
    ok = v.notna()
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    out[ok] = pd.to_datetime(v[ok].astype("int64").astype(str), format="%Y%m%d")
    return out


def from_sackmann(path, alias):
    d = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "source": f"sackmann:{Path(path).name}",
        "tourney_id": d.tourney_id.astype(str), "tourney_name": d.tourney_name, "tourney_slug": pd.NA,
        "level": d.tourney_level, "surface": d.surface.fillna("Hard").str.title().replace({"Carpet": "Hard"}),
        "draw_size": d.draw_size, "best_of": d.best_of,
        "event_start": pd.to_datetime(d.tourney_date.astype(str), format="%Y%m%d"),
        "round": d["round"], "match_num": d.match_num,
        "winner": d.winner_name.map(alias), "loser": d.loser_name.map(alias),
        "winner_rank": d.winner_rank, "loser_rank": d.loser_rank,
        "score": d.score, "minutes": d.minutes,
        # optional columns (TennisMyLife files written by import_tml.py); plain Sackmann files have neither
        "match_date": _ymd(d["match_date"]) if "match_date" in d else pd.NaT,
    })
    for side in ("w", "l"):
        for c in STAT_COLS:
            out[f"{side}_{c}"] = d.get(f"{side}_{c}", np.nan)
    out["event_end"] = _ymd(d["event_end"]) if "event_end" in d else pd.NaT
    return out


def from_courtiq(path, alias, configs):
    d = pd.read_csv(path)
    missing = [c for c in TEMPLATE_COLUMNS[:7] if c not in d.columns]
    if missing: sys.exit(f"{path}: missing required columns {missing}")
    rows = []
    for _, r in d.iterrows():
        m = re.match(r"^([a-z]+)(\d{4})$", str(r.tourney_slug).lower())
        if not m or m.group(1) not in configs: sys.exit(f"{path}: unknown tourney_slug {r.tourney_slug}")
        cfg = configs[m.group(1)]
        key = m.group(1)
        if cfg["level"] == "G": draw = 128
        elif cfg["level"] == "M": draw = 56 if key in ("montecarlo", "paris") else 96   # 12-day Masters since 2025
        elif cfg["level"] == "F": draw = 8
        else: draw = {"R32": 32, "R16": 16}.get(cfg["rounds"][0], 32)
        rows.append({
            "source": f"courtiq:{Path(path).name}", "tourney_id": str(r.tourney_slug).lower(), "tourney_name": f"{cfg['name']} {m.group(2)}",
            "tourney_slug": str(r.tourney_slug).lower(), "level": {"A": "A"}.get(cfg["level"], cfg["level"]),
            "surface": cfg["surface"], "draw_size": draw, "best_of": cfg["best_of"],
            "event_start": pd.to_datetime(r.event_start), "round": r["round"], "match_num": r.match_no,
            "winner": alias(r.winner), "loser": alias(r.loser), "winner_rank": r.get("winner_rank", np.nan), "loser_rank": r.get("loser_rank", np.nan),
            "score": r.score, "minutes": r.get("minutes", np.nan), "match_date": pd.to_datetime(r.get("match_date", pd.NaT)),
            "event_end": pd.to_datetime(r.get("event_end", pd.NaT)),
            **{f"{s}_{c}": r.get(f"{s}_{c}", np.nan) for s in ("w", "l") for c in STAT_COLS},
        })
    return pd.DataFrame(rows)


def finalize(df):
    df = df.copy()
    parsed = df.score.map(parse_score)
    df[["games_w", "games_l", "sets_w", "sets_l", "retired", "walkover"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(3)
    # Sean-supplied rows (source courtiq:*) usually lack minutes. Estimate from games played so the
    # fatigue features don't read them as 0 minutes. Fit on 2021-25 Sackmann rows with real minutes:
    # Bo3 -8.6 + 4.99/game (typical error 11 min), Bo5 -9.0 + 4.71/game (16 min), retirements 4.88/game.
    # Sackmann rows are left exactly as they were.
    games = df.games_w + df.games_l
    est = np.where(df.retired, 4.88 * games,
          np.where(df.best_of == 5, -8.98 + 4.71 * games, -8.64 + 4.99 * games)).clip(min=0)
    fill = df.source.astype(str).str.startswith("courtiq:") & df.minutes.isna() & ~df.walkover & games.notna()
    df["minutes_est"] = fill
    df.loc[fill, "minutes"] = est[fill.values].round()
    dur = [event_duration_days(l, s, b) for l, s, b in zip(df.level, df.draw_size, df.best_of)]
    est_end = df.event_start + pd.to_timedelta(dur, unit="D")
    df["event_end"] = pd.to_datetime(df["event_end"]).fillna(est_end) if "event_end" in df else est_end   # explicit end date wins
    est = df.event_start + pd.to_timedelta([est_match_offset(r, s, l) for r, s, l in zip(df["round"], df.draw_size, df.level)], unit="D")
    df["match_date_est"] = df.match_date.fillna(est)
    df["match_id"] = df.tourney_id + "-" + df["round"].astype(str) + "-" + df.match_num.astype(str)
    df = df.sort_values(["event_start", "tourney_id", "round_order", "match_num"], kind="mergesort").reset_index(drop=True)
    return df


def validate(df):
    problems = []
    def chk(mask, label):
        n = int(mask.sum())
        print(f"  {'OK ' if n == 0 else 'WARN'} {label}: {n}")
        if n: problems.append((label, df[mask].head(5)))
    print("Validation")
    chk(df.duplicated("match_id"), "duplicate match_id")
    chk(df.duplicated(["tourney_id", "winner", "loser", "round"]), "duplicate pairing within an event and round")
    chk(df.winner == df.loser, "winner equals loser")
    chk(df.winner.isna() | df.loser.isna(), "missing player name")
    chk(df.games_w.isna() & ~df.walkover, "score not parseable (not a walkover)")
    chk(~df.walkover & ~df.retired & (df.sets_w <= df.sets_l), "completed match where winner didn't win more sets")
    chk(df.match_date_est > df.event_end, "estimated match date after event end")
    print(f"  matches {len(df):,} | events {df.tourney_id.nunique()} | players {pd.unique(df[['winner','loser']].values.ravel()).size:,}")
    print(f"  with serve stats {df.w_svpt.notna().mean():.1%} | with minutes {df.minutes.notna().mean():.1%} | walkovers {df.walkover.sum()} | retirements {df.retired.sum()}")
    by = df.groupby(df.event_start.dt.to_period("Q")).size()
    print("  matches by quarter:", ", ".join(f"{k}:{v}" for k, v in by.items()))
    for label, ex in problems:
        print(f"\n  Examples - {label}:\n{ex[['match_id','event_start','round','winner','loser','score']].to_string(index=False)}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="CourtIQ repo root (for ALIASES and TOURNEY_CONFIGS)")
    ap.add_argument("--sackmann", nargs="*", default=[])
    ap.add_argument("--courtiq", nargs="*", default=[])
    ap.add_argument("--out", default="data/v2/matches.csv")
    a = ap.parse_args()
    aliases, configs = load_repo_maps(Path(a.repo))
    alias = make_alias(aliases)
    parts = [from_sackmann(p, alias) for p in a.sackmann] + [from_courtiq(p, alias, configs) for p in a.courtiq]
    df = finalize(pd.concat(parts, ignore_index=True))
    # one source per event: if an event appears in two inputs, keep the first input listed
    first_src = df.groupby("tourney_id").source.first()
    df = df[df.source == df.tourney_id.map(first_src)].reset_index(drop=True)
    validate(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nWrote {a.out}")
