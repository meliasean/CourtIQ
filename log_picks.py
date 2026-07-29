#!/usr/bin/env python3
"""
Log CCK Pick Score picks for a tournament round, BEFORE matches play.

Usage:
    python log_picks.py <tournament> <round> [reports_dir]

    e.g. python log_picks.py rg2026 R64
         python log_picks.py rg2026 QF

Reads:   reports/<tournament>_<round>_predictions_cck.csv  (non-complete; pre-match)
Appends to: live_log/live_picks_log.csv

Behavior:
  - Each match becomes one row with the locked CCK pick, all 4 sub-scores, the
    Pick Score, the bucket, plus raw inputs needed for later audit.
  - Voids only happen on opponent swap (same tourney/round/match_no, different
    players) — old row gets voided=true, new row appended.
  - Idempotent: re-running with the same matchups does nothing.
  - Walkovers (rows without odds) are skipped — no log row created.

Formula: same as backtest_pickscore.py, locked. No tuning.
"""
import sys, os, csv
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Reuse the backtest's locked sub-score functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_pickscore import (
    edge_score, confidence_score, agreement_score, volatility_score,
    fair_prob_pair, american_to_profit_per_unit, categorize,
)

LOG_DIR = "live_log"
LOG_PATH = os.path.join(LOG_DIR, "live_picks_log.csv")

LOG_COLUMNS = [
    "logged_at", "tourney", "round", "match_no",
    "player_a", "player_b",
    "pick_player", "pick_odds", "is_underdog",
    "model_pick_prob", "other_model_pick_prob", "book_pick_prob",
    "raw_edge", "edge_score", "confidence_score", "agreement_score", "volatility_score",
    "pick_score", "courtiq_score", "bucket",
    "voided", "voided_at", "voided_reason",
    "outcome", "profit_units", "updated_at",
]

def ensure_log_exists():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()

def load_log():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)

def write_log(df):
    df = df[LOG_COLUMNS]  # canonical column order
    df.to_csv(LOG_PATH, index=False)

def main():
    if len(sys.argv) < 3:
        print("Usage: python log_picks.py <tournament> <round> [reports_dir]")
        sys.exit(1)
    tourney, rnd = sys.argv[1], sys.argv[2]
    reports = sys.argv[3] if len(sys.argv) > 3 else "reports"

    # Pre-match files: non-complete CCK + non-complete standard (for agreement/disagreement)
    cck_path = os.path.join(reports, f"{tourney}_{rnd}_predictions_cck.csv")
    std_path = os.path.join(reports, f"{tourney}_{rnd}_predictions.csv")
    if not os.path.exists(cck_path):
        print(f"ERROR: {cck_path} not found. Run predict first."); sys.exit(1)
    if not os.path.exists(std_path):
        print(f"ERROR: {std_path} not found. Need both files for agreement scoring."); sys.exit(1)
    cck_df = pd.read_csv(cck_path)
    std_df = pd.read_csv(std_path)
    if len(cck_df) != len(std_df):
        print(f"ERROR: row mismatch — cck {len(cck_df)} vs std {len(std_df)}"); sys.exit(1)

    ensure_log_exists()
    log = load_log()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    new_rows, voided_count, skipped_walkover, skipped_already = 0, 0, 0, 0

    for i, r in cck_df.iterrows():
        match_no = i + 1
        pa, pb = str(r["player_a"]), str(r["player_b"])
        oa, ob = r.get("odds_player_a"), r.get("odds_player_b")
        from backtest_pickscore import _valid_american
        if not (_valid_american(oa) and _valid_american(ob)):
            skipped_walkover += 1; continue

        # Pick the player CCK favors
        pA_cck = r.get("prob_player_a_win")
        pA_std = std_df.iloc[i].get("prob_player_a_win")
        if pd.isna(pA_cck) or pd.isna(pA_std):
            print(f"  WARN match {match_no} {pa} vs {pb}: missing prob, skipped"); continue
        cck_pick_a = pA_cck >= 0.50
        std_pick_a = pA_std >= 0.50
        # Devigged book prob
        fa, fb = fair_prob_pair(oa, ob)
        if fa is None:
            print(f"  WARN match {match_no} {pa} vs {pb}: can't compute fair prob, skipped"); continue
        book_pick_a = fa >= 0.50

        pick_player = pa if cck_pick_a else pb
        pick_prob = pA_cck if cck_pick_a else (1 - pA_cck)
        other_prob = pA_std if cck_pick_a else (1 - pA_std)
        book_pick_prob = fa if cck_pick_a else fb
        pick_odds = float(oa if cck_pick_a else ob)
        is_underdog = pick_odds > 0

        ed_s, raw_edge = edge_score(pick_prob, book_pick_prob)
        co_s = confidence_score(pick_prob)
        ag_s = agreement_score(cck_pick_a, std_pick_a, book_pick_a)
        vo_s = volatility_score(co_s, book_pick_prob, pick_prob, other_prob, pick_odds)
        ps = 0.45*ed_s + 0.30*co_s + 0.15*ag_s - 0.10*vo_s
        score_100 = 100 * ps
        bucket = categorize(score_100, raw_edge, vo_s, co_s, is_underdog)

        # Check existing log rows for same tourney/round/match_no
        existing = log[
            (log["tourney"] == tourney) & (log["round"] == rnd) &
            (log["match_no"] == match_no) & (log["voided"] != True)
        ]
        if len(existing) > 0:
            # Same matchup? Then it's already logged, skip.
            prev = existing.iloc[-1]
            same_pair = (
                (str(prev["player_a"]) == pa and str(prev["player_b"]) == pb) or
                (str(prev["player_a"]) == pb and str(prev["player_b"]) == pa)
            )
            if same_pair:
                # Same matchup — check if pick direction flipped (odds shift)
                if str(prev["pick_player"]) == pick_player:
                    skipped_already += 1; continue
                # Pick direction changed → fall through to void + re-log
                void_mask_flip = (
                    (log["tourney"] == tourney) & (log["round"] == rnd) &
                    (log["match_no"] == match_no) & (log["voided"] != True)
                )
                log.loc[void_mask_flip, "voided"] = True
                log.loc[void_mask_flip, "voided_at"] = now
                log.loc[void_mask_flip, "voided_reason"] = f"odds shift changed model pick → {pick_player}"
                voided_count += int(void_mask_flip.sum())
            # Different matchup → void all prior live rows for this match_no
            void_mask = (
                (log["tourney"] == tourney) & (log["round"] == rnd) &
                (log["match_no"] == match_no) & (log["voided"] != True)
            )
            log.loc[void_mask, "voided"] = True
            log.loc[void_mask, "voided_at"] = now
            log.loc[void_mask, "voided_reason"] = f"opponent swap → {pa} vs {pb}"
            voided_count += int(void_mask.sum())

        # Append new row
        new = {
            "logged_at": now,
            "tourney": tourney, "round": rnd, "match_no": match_no,
            "player_a": pa, "player_b": pb,
            "pick_player": pick_player, "pick_odds": pick_odds, "is_underdog": is_underdog,
            "model_pick_prob": round(pick_prob, 6),
            "other_model_pick_prob": round(other_prob, 6),
            "book_pick_prob": round(book_pick_prob, 6),
            "raw_edge": round(raw_edge, 6),
            "edge_score": round(ed_s, 4),
            "confidence_score": round(co_s, 4),
            "agreement_score": round(ag_s, 4),
            "volatility_score": round(vo_s, 4),
            "pick_score": round(ps, 4),
            "courtiq_score": round(score_100, 2),
            "bucket": bucket,
            "voided": False, "voided_at": "", "voided_reason": "",
            "outcome": "", "profit_units": "", "updated_at": "",
        }
        log = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
        new_rows += 1

    write_log(log)
    print(f"\n{tourney} {rnd}:  {new_rows} new picks logged, {voided_count} prior rows voided, "
          f"{skipped_already} unchanged, {skipped_walkover} walkovers skipped")
    if new_rows:
        # Show bucket distribution for this round
        latest = log[(log["tourney"]==tourney) & (log["round"]==rnd) & (log["voided"]!=True)]
        print("\nBucket distribution this round:")
        for b in ["Top Pick","Green","Upset Watch","Lean","Avoid"]:
            n = (latest["bucket"]==b).sum()
            if n: print(f"  {b:<14} {n}")

if __name__ == "__main__":
    main()
