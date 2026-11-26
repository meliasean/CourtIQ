"""Bring TennisMyLife's 2026 season into the V2 store (research use only; see licence note in README).

  python v2/import_tml.py --repo . --tml data/v2/raw/tml_2026.csv --out-dir data/v2/results_2026/tml
         [--check data/v2/results_2026/ao2026.csv ...] [--allow-missing]

Two outputs:
  <slug>.csv            one per event Sean logged (reports/<slug>_*_predictions_complete.csv exists).
                        Template format, CourtIQ names and match_no from the prediction files, plus TML's
                        serve stats, minutes, ranks and real match dates; event_end = date of the final.
                        Main-draw TML matches Sean didn't log (e.g. an early round) are added with
                        match_no 900+ so the profiles still see them.
  unlogged_2026.csv     every other 2026 tour-level event, in Sackmann format, names mapped to the
                        store's spellings. Same event types as the 2021-25 store (United Cup excluded).

Checks (exit non-zero and write nothing on a hard failure):
  - each logged event maps to exactly one TML event (by player overlap and dates)
  - every prediction row matches exactly one TML match in that round
  - TML's winner agrees with the tracker's recorded result (disagreements are listed, not fatal:
    they're either tracker errors or TML errors, and V1 files are never touched)
  - --check: page-built files (import_results_page.py) must agree with TML on winner and score
"""
import argparse, re, sys, unicodedata


def score_key(x):
    """compare scores ignoring tiebreak points, RET/W/O markers and a 0-0 set at a retirement"""
    x = re.sub(r"\(\d+\)|RET|W/O|DEF", "", str(x))
    return " ".join(t for t in x.split() if t != "0-0")

from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_store import load_repo_maps, make_alias, STAT_COLS   # noqa: E402

STATS = [f"{s}_{c}" for s in ("w", "l") for c in STAT_COLS]
LEVEL = {"250": "A", "500": "A"}
EXCLUDE = {"United Cup"}                     # not in the 2021-25 store


def norm(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


def tokens(s):
    return tuple(sorted(norm(t) for t in re.split(r"[\s\-']+", unicodedata.normalize("NFKD", str(s))) if norm(t)))


def same_player(a, b):
    if norm(a) == norm(b) or tokens(a) == tokens(b): return True
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb: return False
    # surname containment with matching first initial, e.g. "Felix Auger-Aliassime" / "Felix Auger Aliassime"
    la, lb = norm(a.split()[-1]), norm(b.split()[-1])
    return (la in norm(b) or lb in norm(a)) and norm(a)[0] == norm(b)[0]


def ymd(x):
    return pd.to_datetime(str(int(x)), format="%Y%m%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--tml", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--allow-missing", action="store_true", help="testing only: tolerate prediction rows TML lacks")
    a = ap.parse_args()
    repo = Path(a.repo)
    aliases, _ = load_repo_maps(repo)
    alias = make_alias(aliases)

    t = pd.concat([pd.read_csv(f, low_memory=False) for f in a.tml], ignore_index=True)
    t = t.drop_duplicates(subset=["tourney_id", "round", "winner_name", "loser_name"])
    t = t[~t.tourney_name.isin(EXCLUDE) & ~t["round"].astype(str).str.startswith("Q")].copy()
    t["tourney_level"] = t.tourney_level.astype(str).replace(LEVEL)
    t["date"] = t.tourney_date.map(ymd)
    per_match = t.groupby("tourney_id").date.transform("nunique") > 1
    t["ev_start"] = t.groupby("tourney_id").date.transform("min")
    t["ev_end"] = t.groupby("tourney_id").date.transform("max").where(per_match)       # real final date when known
    t["m_date"] = t.date.where(per_match)
    print(f"TML: {len(t)} main-draw matches, {t.tourney_id.nunique()} events, "
          f"{t.date.min():%Y-%m-%d} to {t.date.max():%Y-%m-%d}, per-match dates for {per_match.mean():.0%} of rows")

    # known spellings: 2021-25 store + every prediction file
    store = pd.read_csv(repo / "data/v2/matches.csv", usecols=["winner", "loser", "event_start"], low_memory=False)
    known = set(store.winner) | set(store.loser)
    preds = {}
    for f in sorted((repo / "reports").glob("*2026_*_predictions_complete.csv")):
        if "_cck" in f.name: continue
        slug = f.name.split("_")[0]
        preds.setdefault(slug, []).append(pd.read_csv(f))
    preds = {k: pd.concat(v, ignore_index=True) for k, v in preds.items()}
    for d in preds.values(): known |= set(d.player_a) | set(d.player_b)
    by_norm, by_tok = {}, {}
    for n in known: by_norm.setdefault(norm(n), n); by_tok.setdefault(tokens(n), n)

    def canon(name):
        n = alias(name)
        if n in known: return n, "same"
        for key, table in ((norm(n), by_norm), (tokens(n), by_tok)):
            if key in table: return table[key], "normalised"
        return n, "new"

    hard, tracker, notes, logged_ids = [], [], [], {}
    os_ = Path(a.out_dir); written = {}
    # ---- logged events
    for slug, p in sorted(preds.items()):
        names = set(p.player_a) | set(p.player_b)
        lo, hi = pd.to_datetime(p.date, format="mixed").min() - pd.Timedelta(days=12), pd.to_datetime(p.date, format="mixed").max() + pd.Timedelta(days=12)
        cand = []
        for tid, g in t[(t.date >= lo) & (t.date <= hi)].groupby("tourney_id"):
            tn = set(g.winner_name) | set(g.loser_name)
            hit = sum(any(same_player(x, y) for y in tn) for x in names)
            cand.append((hit / len(names), tid, g.tourney_name.iloc[0]))
        cand.sort(reverse=True)
        if not cand or cand[0][0] < 0.6 or (len(cand) > 1 and cand[1][0] > 0.6):
            if a.allow_missing: notes.append(f"{slug}: no TML event found (skipped)"); continue
            hard.append(f"{slug}: can't pin a single TML event (best {cand[:2]})"); continue
        tid = cand[0][1]; logged_ids[tid] = slug
        g = t[t.tourney_id == tid]
        used, rows = set(), []
        for _, x in p.iterrows():
            rr = "RR" if str(x["round"]).startswith("RR") else x["round"]
            m = [i for i, y in g[g["round"] == rr].iterrows() if i not in used and
                 ((same_player(x.player_a, y.winner_name) and same_player(x.player_b, y.loser_name)) or
                  (same_player(x.player_b, y.winner_name) and same_player(x.player_a, y.loser_name)))]
            if len(m) != 1:
                (notes if a.allow_missing else hard).append(f"{slug} {x['round']} match_no {x.match_no}: {x.player_a} v {x.player_b} -> {len(m)} TML rows")
                continue
            used.add(m[0]); y = t.loc[m[0]]
            a_won = same_player(x.player_a, y.winner_name)
            if pd.notna(x.correct_prediction):
                rec_a = (x.pred_winner == x.player_a) == (int(x.correct_prediction) == 1)
                if rec_a != a_won:
                    tracker.append(f"{slug} {x['round']} match_no {x.match_no}: tracker has {x.player_a if rec_a else x.player_b}, "
                                   f"TML has {y.winner_name} ({y.score})")
            rows.append({"tourney_slug": slug, "round": x["round"], "match_no": int(x.match_no),
                         "winner": x.player_a if a_won else x.player_b, "loser": x.player_b if a_won else x.player_a, "_y": y})
        extra = g.drop(index=list(used))
        for k, (_, y) in enumerate(extra.iterrows()):
            w, l = canon(y.winner_name)[0], canon(y.loser_name)[0]
            rows.append({"tourney_slug": slug, "round": y["round"], "match_no": 900 + k, "winner": w, "loser": l, "_y": y})
        if len(extra): notes.append(f"{slug}: {len(extra)} TML main-draw matches not in your prediction files, added as 900+ "
                                    f"({extra['round'].value_counts().to_dict()})")
        out = pd.DataFrame([{**{k: v for k, v in r.items() if k != "_y"},
                             "event_start": r["_y"].ev_start.strftime("%Y-%m-%d"), "score": r["_y"].score,
                             "minutes": r["_y"].minutes,
                             "match_date": "" if pd.isna(r["_y"].m_date) else r["_y"].m_date.strftime("%Y-%m-%d"),
                             "event_end": "" if pd.isna(r["_y"].ev_end) else r["_y"].ev_end.strftime("%Y-%m-%d"),
                             "winner_rank": r["_y"].winner_rank, "loser_rank": r["_y"].loser_rank,
                             **{c: r["_y"].get(c, np.nan) for c in STATS}} for r in rows])
        cols = ["tourney_slug", "event_start", "round", "match_no", "winner", "loser", "score", "minutes",
                "match_date", "event_end", "winner_rank", "loser_rank"] + STATS
        written[f"{slug}.csv"] = out[cols].assign(score_source="tml")

    # ---- page-built cross-checks
    for f in a.check:
        c = pd.read_csv(f); slug = c.tourney_slug.iloc[0]
        if f"{slug}.csv" not in written: hard.append(f"--check {f}: no TML output for {slug}"); continue
        out = written[f"{slug}.csv"]
        m = c.merge(out.reset_index(), on=["round", "match_no"], suffixes=("_page", "_tml"))
        wrong_w = m[m.winner_page != m.winner_tml]
        diff = m[(m.winner_page == m.winner_tml) & (m.score_page.map(score_key) != m.score_tml.map(score_key))]
        print(f"check {slug}: {len(m)} matches compared with the page-built file: {len(wrong_w)} winner differences, "
              f"{len(diff)} score differences (page score kept)")
        if len(diff): print(diff[["round", "match_no", "winner_page", "score_page", "score_tml"]].to_string(index=False))
        out.loc[m["index"], "score"] = m.score_page.values          # page score wins wherever both exist
        out.loc[m["index"], "score_source"] = "page"
        if len(wrong_w): hard.append(f"{slug}: page and TML disagree on the winner:\n{wrong_w[['round','match_no','winner_page','winner_tml']].to_string(index=False)}")

    # ---- unlogged events
    u = t[~t.tourney_id.isin(logged_ids)].copy()
    newp = {}
    for side in ("winner", "loser"):
        res = u[f"{side}_name"].map(canon)
        u[f"{side}_name"] = res.map(lambda r: r[0])
        for n, how in res:
            if how == "new": newp[n] = newp.get(n, 0) + 1
    u["tourney_date"] = u.ev_start.dt.strftime("%Y%m%d").astype(int)
    u["match_date"] = u.m_date.dt.strftime("%Y%m%d")
    u["event_end"] = u.ev_end.dt.strftime("%Y%m%d")
    keep = [c for c in pd.read_csv(a.tml[0], nrows=0).columns] + ["match_date", "event_end"]
    written["unlogged_2026.csv"] = u[keep]

    print(f"\nlogged events found in TML: {len(logged_ids)} of {len(preds)}")
    for tid, slug in sorted(logged_ids.items(), key=lambda kv: kv[1]):
        g = t[t.tourney_id == tid]
        print(f"  {slug:<16} -> {tid:<10} {g.tourney_name.iloc[0]:<24} {len(written[slug + '.csv']):>4} matches, "
              f"stats {written[slug + '.csv'].w_svpt.notna().mean():.0%}")
    print(f"unlogged events: {u.tourney_id.nunique()} ({len(u)} matches), levels {u.tourney_level.value_counts().to_dict()}")
    print(f"players new to the store (no 2021-25 match, no prediction-file spelling): {len(newp)}")
    if newp: print("  " + ", ".join(f"{k} ({v})" for k, v in sorted(newp.items(), key=lambda kv: -kv[1])[:40]))
    for n in notes: print("note: " + n)
    if tracker: print(f"\nTRACKER DISAGREES WITH TML on {len(tracker)} matches (V1 files not touched):\n  " + "\n  ".join(tracker))
    if hard:
        print("\nPROBLEMS - nothing written:\n  " + "\n  ".join(hard)); sys.exit(1)
    os_.mkdir(parents=True, exist_ok=True)
    for k, v in written.items(): v.to_csv(os_ / k, index=False)
    print(f"\nwrote {len(written)} files to {os_}")


if __name__ == "__main__":
    main()
