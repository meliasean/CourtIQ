"""
generate_rome2026.py
=====================
Rome 2026 — Internazionali BNL d'Italia — ATP Masters 1000, Clay
Generates R128 prediction files in correct bracket order.

96-player draw: top 32 seeds have byes to R64.
R128 = 32 matches between unseeded/lower-seeded players.
Matches listed in bracket order (top to bottom) as they appear in the draw.

Run from your project root:
    python generate_rome2026.py
"""

import unicodedata, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

warnings.filterwarnings("ignore")

REPORTS_DIR  = Path("./reports")
MODEL_PATHS  = ["./models/rf_model.joblib"]
CAL_PATH     = Path("./models/prob_calibrator.joblib")
PROFILE_PATHS = ["./reports/player_profiles_latest.csv",
                 "./reports/player_profiles_post_madrid_2026.csv"]
SURFACE = "Clay"; LEVEL = "M"; BEST_OF = 3; SLUG = "rome2026"

def alias(n):
    if not n or pd.isna(n): return n
    return unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii").strip()

def ap(o):
    if o is None or pd.isna(o): return np.nan
    try: o = float(o)
    except: return np.nan
    return (-o)/((-o)+100) if o < 0 else 100/(o+100)

def devig(a, b):
    if any(np.isnan(v) for v in [a,b]) or a<=0 or b<=0: return np.nan, np.nan
    s = a+b; return a/s, b/s

def logit(p): return np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def sigmoid(z): return 1/(1+np.exp(-z))
def elo_p(ea, eb): return 1/(1+10**(-(ea-eb)/400))

def load_profiles():
    for p in PROFILE_PATHS:
        if Path(p).exists():
            df = pd.read_csv(p)
            df["name"] = df["name"].astype(str).str.strip()
            print(f"  Profiles: {p} ({len(df)} players)")
            return df
    raise FileNotFoundError("No profiles found")

def snap(prof, player):
    player = alias(player)
    rows = prof[prof["name"] == player]
    D = {"pre_elo":1500,"selo_Clay":1500,"avg_rest_days":20,"matches_28d":0,
         "form10_wr":.5,"form5_wr":.5,"streak":0,"overall_wr":.5,"peak_elo":1500,
         "current_elo":1500,"wr_Clay":.5,"wr_Grass":.5,"wr_Hard":.5,
         "h2h_wr_prior":.5,"cnt_Clay":0}
    if rows.empty:
        print(f"    WARN: '{player}' not in profiles")
        return D
    r = rows.iloc[-1]
    def g(c, d=0):
        try: return float(r.get(c,d) or d)
        except: return d
    return {**D,
        "pre_elo":g("current_elo",1500), "current_elo":g("current_elo",1500),
        "peak_elo":g("peak_elo",1500),   "selo_Clay":g("selo_Clay",1500),
        "overall_wr":g("overall_wr",.5), "avg_rest_days":g("avg_rest_days",20),
        "matches_28d":g("matches_28d",0),"form10_wr":g("form10_wr",.5),
        "form5_wr":g("form5_wr",.5),     "streak":g("streak",0),
        "wr_Clay":g("wr_Clay",.5),       "wr_Grass":g("wr_Grass",.5),
        "wr_Hard":g("wr_Hard",.5),       "h2h_wr_prior":g("h2h_wr_prior",.5),
        "cnt_Clay":g("cnt_Clay",0)}

BF = ["pre_elo","selo_Clay","avg_rest_days","matches_28d","form10_wr","form5_wr",
      "streak","overall_wr","peak_elo","current_elo","wr_Clay","wr_Grass","wr_Hard",
      "h2h_wr_prior","cnt_Clay"]
BFMAP = {"selo_Clay":"pre_selo","form10_wr":"rolling_10_winrate",
         "form5_wr":"rolling_5_winrate","overall_wr":"win_rate"}

def d(sa, sb, k): return float(sa.get(k,0) or 0) - float(sb.get(k,0) or 0)

def predict_match(pa, pb, oa, ob, rnd, pipe, fcols, calibrator, prof, date, winner):
    sa = snap(prof, pa); sb = snap(prof, pb)
    feats = {"surface":SURFACE,"tourney_level":LEVEL,"round":rnd,"best_of":BEST_OF}
    for f in BF:
        key = BFMAP.get(f, f)
        feats[f"diff_{key}"] = d(sa, sb, f)
    feats["elo_diff"]  = d(sa, sb, "pre_elo")
    feats["selo_diff"] = d(sa, sb, "selo_Clay")
    feats["rank_diff"] = 0.0
    fd = pd.DataFrame([feats])
    for c in fcols:
        if c not in fd.columns: fd[c] = feats.get(c, 0)
    p_raw = float(pipe.predict_proba(fd[fcols])[:,1][0])
    if calibrator is not None:
        try: p_cal = float(calibrator.predict([p_raw])[0])
        except: p_cal = p_raw
    else:
        p_cal = p_raw
    pred = pa if p_cal >= 0.5 else pb
    conf = max(p_cal, 1-p_cal)
    paf, pbf = devig(ap(oa), ap(ob))
    p_elo  = elo_p(sa["pre_elo"], sb["pre_elo"])
    p_temp = sigmoid(logit(p_cal) / 1.30)
    cp = np.nan; cpb = np.nan
    if winner:
        aw = alias(winner)
        cp = 1 if alias(pred) == aw else 0
        if not np.isnan(paf):
            bk = pa if paf >= 0.5 else pb
            cpb = 1 if alias(bk) == aw else 0
    return {
        "date": date, "round": rnd, "surface": SURFACE,
        "tourney_level": LEVEL, "best_of": BEST_OF,
        "player_a": pa, "player_b": pb,
        "odds_player_a": float(oa) if oa is not None else np.nan,
        "odds_player_b": float(ob) if ob is not None else np.nan,
        "pred_winner": pred, "confidence": round(conf, 6),
        "prob_player_a_win": round(p_cal, 6),
        "prob_player_b_win": round(1-p_cal, 6),
        "book_fair_prob_a": round(paf, 6) if not np.isnan(paf) else np.nan,
        "book_fair_prob_b": round(pbf, 6) if not np.isnan(pbf) else np.nan,
        "p_elo_a": round(p_elo, 6), "p_temp_a": round(p_temp, 6),
        "delta_elo": d(sa, sb, "pre_elo"),
        "correct_prediction": cp,
        "correct_prediction_book": cpb,
    }

def save_round(rows, slug, rnd):
    df = pd.DataFrame(rows)
    df.insert(0, "match_no", range(1, len(df)+1))
    base = REPORTS_DIR / f"{slug}_{rnd}"
    df.to_csv(f"{base}_predictions_cck_complete.csv", index=False)
    df.to_csv(f"{base}_predictions_complete.csv", index=False)
    blank = df.copy()
    blank["correct_prediction"] = np.nan
    blank["correct_prediction_book"] = np.nan
    blank.to_csv(f"{base}_predictions_cck.csv", index=False)
    blank.to_csv(f"{base}_predictions.csv", index=False)
    scored = df["correct_prediction"].notna().sum()
    total  = len(df)
    acc_rows = df[df["correct_prediction"].notna()]
    acc  = acc_rows["correct_prediction"].mean() if len(acc_rows) else float("nan")
    bacc = acc_rows["correct_prediction_book"].mean() if len(acc_rows) and acc_rows["correct_prediction_book"].notna().any() else float("nan")
    print(f"  {slug} {rnd}: {scored}/{total} scored  model={acc:.0%}  book={bacc:.0%}" if len(acc_rows) else f"  {slug} {rnd}: 0/{total} scored (pending)")


def main():
    print(f"\n=== Generate {SLUG} R128 ===\n")

    pipe = None; fcols = None; calibrator = None
    for mp in MODEL_PATHS:
        if Path(mp).exists():
            b = load(mp)
            pipe, fcols = (b["pipeline"], b["feature_cols"]) if isinstance(b,dict) else (b,None)
            print(f"  Model: {mp}"); break
    if CAL_PATH.exists():
        calibrator = load(str(CAL_PATH))
        print(f"  Calibrator loaded")
    prof = load_profiles()

    # ── R128 DRAW IN BRACKET ORDER ────────────────────────────
    # (player_a, player_b, odds_a, odds_b, winner_or_None, date)
    # Listed top-to-bottom as they appear in the draw bracket.
    # player_a = top of match, player_b = bottom of match.
    # winner = None for matches not yet played.
    # Seeds with byes (entering at R64) are noted as comments showing
    # which seed the winner of each match will face.

    R128 = [
        # ── TOP HALF ──────────────────────────────────────────
        # Match 1: winner faces Sinner (1) in R64
        ("Sebastian Ofner",          "Alex Michelsen",           -167,  138, None,                      "2026-05-07"),
        # Match 2: winner faces Mensik (26) in R64
        ("Alexei Popyrin",           "Matteo Berrettini",         175, -208, None,                      "2026-05-07"),
        # Match 3: winner faces Tiafoe (20) in R64
        ("Lorenzo Sonego",           "Ignacio Buse",              138, -167, None,                      "2026-05-07"),
        # Match 4: winner faces Fils (15) in R64
        ("Andrea Pellegrino",        "Luca Nardi",               -149,  129, None,                      "2026-05-07"),
        # Match 5: winner faces Rublev (12) in R64
        ("Dalibor Svrcina",          "Miomir Kecmanovic",        -101, -110, None,                      "2026-05-07"),
        # Match 6: winner faces Davidovich Fokina (21) in R64
        ("Cristian Garin",           "Juan Manuel Cerundolo",    -133,  114, None,                      "2026-05-07"),
        # Match 7: winner faces Nakashima (30) in R64
        ("Roberto Bautista Agut",    "Francesco Maestrelli",     -189,  160, None,                      "2026-05-07"),
        # Match 8: winner faces Shelton (5) in R64
        ("Nikoloz Basilashvili",     "Daniel Merida",             150, -167, None,                      "2026-05-07"),
        # Match 9: winner faces Felix Auger-Aliassime (4) in R64
        ("Denis Shapovalov",         "Mariano Navone",            168, -189, None,                      "2026-05-07"),
        # Match 10: winner faces Fonseca (27) in R64
        ("Hamad Medjedovic",         "Valentin Royer",           -278,  240, None,                      "2026-05-07"),
        # Match 11: winner faces Etcheverry (24) in R64
        ("Roman Burruchaga",         "Mattia Bellucci",          -217,  175, None,                      "2026-05-07"),
        # Match 12: winner faces Vacherot (14) in R64
        ("Marcos Giron",             "Marin Cilic",               131, -161, None,                      "2026-05-07"),
        # Match 13: winner faces Cobolli (10) in R64
        ("Zizou Bergs",              "Terence Atmane",           -137,  114, None,                      "2026-05-07"),
        # Match 14: winner faces Norrie (17) in R64
        ("Thiago Agustin Tirante",   "Gianluca Cadenasso",       -185,  163, None,                      "2026-05-07"),
        # Match 15: winner faces Moutet (28) in R64
        ("Ethan Quinn",              "Pablo Llamas Ruiz",         120, -137, None,                      "2026-05-07"),
        # Match 16: winner faces Medvedev (7) in R64
        ("Stefanos Tsitsipas",       "Tomas Machac",             -137,  122, None,                      "2026-05-07"),

        # ── BOTTOM HALF ───────────────────────────────────────
        # Match 17: winner faces Musetti (8) in R64
        ("Jacob Fearnley",           "Giovanni Mpetshi Perricard",-167,  143, None,                     "2026-05-06"),
        # Match 18: winner faces Cerundolo F. (25) in R64
        ("Pablo Carreno Busta",      "Alejandro Tabilo",          200, -233, None,                      "2026-05-06"),
        # Match 19: winner faces Lehecka (11) in R64 — COMPLETED
        ("Marco Trungelliti",        "Zachary Svajda",           -250,  250, "Zachary Svajda",           "2026-05-06"),
        # Match 20: winner faces Lehecka (11) in R64 — COMPLETED
        ("Jan-Lennard Struff",       "Francisco Comesana",        175, -208, "Jan-Lennard Struff",      "2026-05-06"),
        # Match 21: winner faces Kovacevic LL / Humbert (31) in R64 — COMPLETED
        ("Camilo Ugo Carabelli",     "Alexander Shevchenko",     -238,  200, "Alexander Shevchenko",    "2026-05-06"),
        # Match 22: winner faces Humbert (31) in R64 — COMPLETED
        ("Alexandre Muller",         "Botic van de Zandschulp",   125, -149, "Botic van de Zandschulp", "2026-05-06"),
        # Match 23: winner faces Humbert (31) in R64 — COMPLETED
        ("Fabian Marozsan",          "Vit Kopriva",              -161,  129, "Vit Kopriva",             "2026-05-06"),
        # Match 24: winner faces Djokovic (3) in R64
        ("Marton Fucsovics",         "Dino Prizmic",              300, -333, None,                      "2026-05-06"),
        # Match 25: winner faces De Minaur (6) in R64 — COMPLETED
        ("Matteo Arnaldi",           "Jaume Munar",              -143,  120, "Matteo Arnaldi",          "2026-05-06"),
        # Match 26: winner faces Jodar (32) in R64 — COMPLETED
        ("Jesper De Jong",           "Nuno Borges",               129, -128, "Nuno Borges",             "2026-05-06"),
        # Match 27: winner faces Tien (19) in R64 — COMPLETED
        ("Damir Dzumhur",            "Adrian Mannarino",         -294,  250, "Damir Dzumhur",           "2026-05-06"),
        # Match 28: winner faces Bublik (9) in R64 — COMPLETED
        ("Jenson Brooksby",          "Sebastian Baez",            300, -345, "Sebastian Baez",           "2026-05-06"),
        # Match 29: winner faces Paul (16) in R64 — COMPLETED
        ("Aleksandar Vukic",         "Patrick Kypson",            278, -345, "Aleksandar Vukic",        "2026-05-06"),
        # Match 30: winner faces Darderi (18) in R64 — COMPLETED
        ("Hubert Hurkacz",           "Yannick Hanfmann",         -222,  177, "Yannick Hanfmann",        "2026-05-06"),
        # Match 31: winner faces Griekspoor (29) in R64
        ("Federico Cina",            "Alexander Blockx",          333, -370, None,                      "2026-05-06"),
        # Match 32: winner faces Zverev (2) in R64 — COMPLETED
        ("Zhang Zhizhen",            "Daniel Altmaier",           138, -169, "Daniel Altmaier",         "2026-05-06"),
    ]

    REPORTS_DIR.mkdir(exist_ok=True)
    rows = [predict_match(pa, pb, oa, ob, "R128", pipe, fcols, calibrator, prof, date, winner)
            for pa, pb, oa, ob, winner, date in R128]
    save_round(rows, SLUG, "R128")

    completed = sum(1 for *_, w, _ in R128 if w)
    pending   = sum(1 for *_, w, _ in R128 if not w)
    print(f"\n  {completed} results filled, {pending} pending")
    print(f"\nRun:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/rome2026_R128_*.csv docs/index.html")
    print(f"  git commit -m 'add: Rome 2026 R128 predictions — {completed}/32 results in'")
    print(f"  git push")


if __name__ == "__main__":
    main()
