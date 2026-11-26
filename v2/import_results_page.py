"""Import one tournament's results from text copied off the ATP results page.

  python v2/import_results_page.py --repo . --slug ao2026 --page data/v2/raw/pages/ao2026.txt \
      --out data/v2/results_2026/ao2026.csv [--anchor "Carlos Alcaraz=1"]

The page text can be either copy format (both seen for AO 2026):
  - "Player-Photo-<id>" then "A. Name (seed)" then one line per set
  - "[A. Name](https://www.atptour.com/en/players/<slug>/<id>/overview) (seed)" then one line per set
Round headers (R128 ... QF, SF, F, "Final", "Round of 16", ...) are optional: a round that
overflows its draw size spills into the next one, so a missing "F" header is fine.

Winners come from who plays in the next round (the page lists players in draw order, so this is
what gets retirements right); the final goes by sets. Each match is then matched by player names
to the CourtIQ prediction file for that round, which supplies the canonical names and match_no and
whose recorded result must agree. Rankings are recovered from V1's delta_rank (pred loser rank minus
pred winner rank, 2000 = V1's missing-rank placeholder), anchored by --anchor, or by a placeholder
player, or by atp_ranks_<slug>.csv. Minutes and match dates are left blank (match_store estimates
them). Exits non-zero if any check fails, and writes nothing.
"""
import argparse, re, sys, unicodedata
from pathlib import Path
import pandas as pd

ORDER = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
SIZE = {"R128": 64, "R64": 32, "R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1}
HEADERS = {"r128": "R128", "roundof128": "R128", "1stround": "R128", "r64": "R64", "roundof64": "R64",
           "r32": "R32", "roundof32": "R32", "r16": "R16", "roundof16": "R16", "qf": "QF", "quarterfinals": "QF",
           "quarterfinal": "QF", "sf": "SF", "semifinals": "SF", "semifinal": "SF", "f": "F", "final": "F", "finals": "F"}
MD = re.compile(r"^\[(?P<name>[^\]]+)\]\(https?://www\.atptour\.com/en/players/(?P<slug>[^/]+)/(?P<id>[^/]+)/overview\)")


def norm(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


def parse_page(text):
    """-> list of (header_or_None, [player, player]) in page order."""
    out, cur, header, lines, i = [], [], None, [l.strip() for l in text.splitlines()], 0
    while i < len(lines):
        l = lines[i]
        m = MD.match(l)
        if l.startswith("Player-Photo-") or m:
            if m: pid, name, i = m["id"], m["name"], i + 1
            else: pid, name, i = l[len("Player-Photo-"):], lines[i + 1], i + 2
            games = []
            while i < len(lines) and re.fullmatch(r"\d+", lines[i]): games.append(lines[i]); i += 1
            cur.append({"id": pid, "name": re.sub(r"\s*\(.*?\)\s*$", "", name).strip(), "games": games})
            if len(cur) == 2: out.append((header, cur)); cur, header = [], None
            continue
        h = HEADERS.get(norm(l))
        if h and not cur: header = h
        i += 1
    if cur: sys.exit(f"page ends mid-match after {cur[0]['name']}")
    return out


def assign_rounds(items, rounds):
    got, k = {r: [] for r in rounds}, 0
    for header, m in items:
        if header: k = rounds.index(header)
        while len(got[rounds[k]]) >= SIZE[rounds[k]]:
            k += 1
            if k == len(rounds): sys.exit("more matches on the page than the draw holds")
        got[rounds[k]].append(m)
    return got


def set_pair(a, b):
    tb = lambda x, o: len(x) >= 2 and x.startswith("6") and o == "7"
    if tb(a, b): return 6, 7, int(a[1:])
    if tb(b, a): return 7, 6, int(b[1:])
    return int(a), int(b), None


def complete(x, y):
    hi, lo = max(x, y), min(x, y)
    return (hi == 6 and lo <= 4) or (hi == 7 and lo in (5, 6))


def surname_key(name):
    s = name.split(". ", 1)[1] if ". " in name else name
    return norm(re.sub(r"\b(Jr|Sr)\.?$", "", s).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--slug", required=True)
    ap.add_argument("--page", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--anchor", help='"Player Name=rank" to pin the ranking scale')
    a = ap.parse_args()
    repo = Path(a.repo)
    preds = {r: pd.read_csv(f) for r in ORDER if (f := repo / f"reports/{a.slug}_{r}_predictions_complete.csv").exists()}
    rounds = [r for r in ORDER if r in preds]
    if not rounds: sys.exit(f"no prediction files for {a.slug}")
    best_of = int(preds[rounds[0]].best_of.iloc[0])
    got = assign_rounds([x for p in a.page for x in parse_page(Path(p).read_text())], rounds)
    problems, rows = [], []
    for r in rounds:
        if len(got[r]) != len(preds[r]): problems.append(f"{r}: page has {len(got[r])} matches, prediction file {len(preds[r])}")
    for k, r in enumerate(rounds):
        nxt = {p["id"] for m in got[rounds[k + 1]] for p in m} if k + 1 < len(rounds) else None
        pred = preds[r]
        for n, (p, q) in enumerate(got[r]):
            hit = [i for i, x in pred.iterrows()
                   if all(surname_key(w["name"]) in norm(x.player_a) + "|" + norm(x.player_b) for w in (p, q))]
            if len(hit) != 1: problems.append(f"{r} #{n+1}: {p['name']} v {q['name']} matched {len(hit)} prediction rows"); continue
            pr = pred.loc[hit[0]]
            if surname_key(p["name"]) not in norm(pr.player_a): p, q = q, p
            sets = [set_pair(x, y) for x, y in zip(p["games"], q["games"])]
            if len(p["games"]) != len(q["games"]): problems.append(f"{r} #{n+1}: uneven set counts")
            sa = sum(ga > gb and complete(ga, gb) for ga, gb, _ in sets)
            sb = sum(gb > ga and complete(ga, gb) for ga, gb, _ in sets)
            if nxt is not None and len(got[rounds[k + 1]]) == SIZE[rounds[k + 1]]:
                on = (p["id"] in nxt, q["id"] in nxt)
                if on[0] == on[1]: problems.append(f"{r} #{n+1}: can't tell who advanced ({p['name']} v {q['name']})"); continue
                a_won = on[0]
            else:
                if sa == sb: problems.append(f"{r} #{n+1}: no next round to check and sets level"); continue
                a_won = sa > sb
            need = best_of // 2 + 1
            if not sets: kind = "W/O"
            elif (sa if a_won else sb) == need and all(complete(ga, gb) for ga, gb, _ in sets): kind = "done"
            else: kind = "RET"
            if kind == "done" and (sa > sb) != a_won: problems.append(f"{r} #{n+1}: sets and next round disagree")
            if pd.notna(pr.correct_prediction):
                rec_a = (pr.pred_winner == pr.player_a) == (int(pr.correct_prediction) == 1)
                if rec_a != a_won:
                    problems.append(f"{r} match_no {int(pr.match_no)}: tracker records {pr.player_a if rec_a else pr.player_b} "
                                    f"as winner, page says {pr.player_a if a_won else pr.player_b}")
            parts = []
            for ga, gb, t in sets:
                w, l = (ga, gb) if a_won else (gb, ga)
                parts.append(f"{w}-{l}" + (f"({t})" if t is not None else ""))
            score = "W/O" if kind == "W/O" else " ".join(parts) + (" RET" if kind == "RET" else "")
            rows.append({"tourney_slug": a.slug, "round": r, "match_no": int(pr.match_no),
                         "winner": pr.player_a if a_won else pr.player_b, "loser": pr.player_b if a_won else pr.player_a,
                         "score": score, "_pred": pr})

    # rankings from delta_rank
    edges = []
    for r in rounds:
        for _, x in preds[r].iterrows():
            w, l = (x.player_a, x.player_b) if x.pred_winner == x.player_a else (x.player_b, x.player_a)
            if pd.notna(x.delta_rank): edges.append((w, l, float(x.delta_rank)))
    def solve(anchor):
        rk, ch = dict([anchor]), True
        while ch:
            ch = False
            for u, v, dr in edges:
                if u in rk and v not in rk: rk[v] = rk[u] + dr; ch = True
                if v in rk and u not in rk: rk[u] = rk[v] - dr; ch = True
        return rk
    anchors = []
    if a.anchor:
        nm, val = a.anchor.rsplit("=", 1); anchors.append(("--anchor", (nm.strip(), float(val))))
    rf = repo / f"atp_ranks_{a.slug}.csv"
    if rf.exists():
        f = pd.read_csv(rf); top = f.sort_values("rank").iloc[0]
        anchors.append((rf.name, (top.player, float(top["rank"]))))
    rel = solve((edges[0][0], 0.0))
    if max(rel.values()) - min(rel.values()) > 1000:   # someone sits on the 2000 placeholder
        hi = max(rel, key=rel.get); anchors.append(("placeholder", (hi, 2000.0)))
    if not anchors: problems.append("no ranking anchor: pass --anchor \"Name=rank\"")
    ranks = {}
    if anchors:
        ranks = solve(anchors[0][1])
        if len(ranks) != len({x for e in edges for x in e[:2]}): problems.append("rank graph not connected")
        bad = [e for e in edges if abs((ranks[e[1]] - ranks[e[0]]) - e[2]) > 1e-6]
        if bad: problems.append(f"{len(bad)} delta_rank rows disagree with each other")
        if min(ranks.values()) < 1 or any(v != int(v) for v in ranks.values()): problems.append("recovered ranks are not positive whole numbers: anchor is off")
        for src, (nm, val) in anchors[1:]:
            if nm in ranks and ranks[nm] != val: problems.append(f"anchors disagree: {anchors[0][0]} puts {nm} at {ranks[nm]}, {src} says {val}")
        print("rank anchors:", ", ".join(f"{s} ({n}={v:g})" for s, (n, v) in anchors))

    first = pd.to_datetime(preds[rounds[0]].date.iloc[0])
    start = (first - pd.Timedelta(days=first.weekday())).strftime("%Y-%m-%d")
    out = pd.DataFrame([{k: v for k, v in x.items() if k != "_pred"} for x in rows])
    out.insert(1, "event_start", start)
    rk = lambda n: ranks.get(n) if ranks.get(n) not in (None, 2000.0) else None
    out["minutes"] = ""; out["match_date"] = ""; out["event_end"] = ""
    out["winner_rank"] = out.winner.map(rk); out["loser_rank"] = out.loser.map(rk)
    out = out.sort_values(["round", "match_no"], key=lambda c: c.map(ORDER.index) if c.name == "round" else c)

    print(f"{a.slug}: {len(out)} matches, event_start {start}, best of {best_of}")
    print(out.groupby("round", sort=False).size().to_string())
    special = out[out.score.str.contains("RET|W/O")]
    if len(special): print("retirements / walkovers:\n" + special[["round", "winner", "loser", "score"]].to_string(index=False))
    print(f"players without a usable rank: {sorted(set(out.winner[out.winner_rank.isna()]) | set(out.loser[out.loser_rank.isna()]))}")
    tracker = [x for x in problems if "tracker records" in x]
    hard = [x for x in problems if x not in tracker]
    if tracker: print("\nTRACKER DISAGREES WITH THE PAGE (V1 files not touched):\n  " + "\n  ".join(tracker))
    if hard:
        print("\nPROBLEMS - nothing written:\n  " + "\n  ".join(hard)); sys.exit(1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
