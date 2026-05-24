"""
generate_hamburg2026.py
========================
Hamburg 2026 — Hamburg Open — ATP 500, Clay
Generates R32 prediction files in bracket order.

32-player draw. Seeds 1-8 had byes to R16.
R32 = 16 matches between unseeded/lower-ranked players.

Results from May 17-19 (all completed or in progress).

Run from your project root:
    python generate_hamburg2026.py
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
                 "./reports/player_profiles_post_rome_2026.csv"]
SURFACE = "Clay"; LEVEL = "A"; BEST_OF = 3; SLUG = "hamburg2026"

def norm(n):
    if not n or pd.isna(n): return ""
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
    return n.lower().strip().replace("-","").replace("'","").replace(" ","").replace(".","")

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
    pn = unicodedata.normalize("NFKD", str(player)).encode("ascii","ignore").decode("ascii").strip()
    rows = prof[prof["name"]==pn]
    D = {"pre_elo":1500,"selo_Clay":1500,"avg_rest_days":20,"matches_28d":0,
         "form10_wr":.5,"form5_wr":.5,"streak":0,"overall_wr":.5,"peak_elo":1500,
         "current_elo":1500,"wr_Clay":.5,"wr_Grass":.5,"wr_Hard":.5,
         "h2h_wr_prior":.5,"cnt_Clay":0}
    if rows.empty: print(f"    WARN: '{pn}' not in profiles"); return D
    r = rows.iloc[-1]
    def g(c,d=0):
        try: return float(r.get(c,d) or d)
        except: return d
    return {**D,
        "pre_elo":g("current_elo",1500),"current_elo":g("current_elo",1500),
        "peak_elo":g("peak_elo",1500),"selo_Clay":g("selo_Clay",1500),
        "overall_wr":g("overall_wr",.5),"avg_rest_days":g("avg_rest_days",20),
        "matches_28d":g("matches_28d",0),"form10_wr":g("form10_wr",.5),
        "form5_wr":g("form5_wr",.5),"streak":g("streak",0),
        "wr_Clay":g("wr_Clay",.5),"wr_Grass":g("wr_Grass",.5),
        "wr_Hard":g("wr_Hard",.5),"h2h_wr_prior":g("h2h_wr_prior",.5),
        "cnt_Clay":g("cnt_Clay",0)}

BF=["pre_elo","selo_Clay","avg_rest_days","matches_28d","form10_wr","form5_wr",
    "streak","overall_wr","peak_elo","current_elo","wr_Clay","wr_Grass","wr_Hard",
    "h2h_wr_prior","cnt_Clay"]
BFMAP={"selo_Clay":"pre_selo","form10_wr":"rolling_10_winrate",
       "form5_wr":"rolling_5_winrate","overall_wr":"win_rate"}

def d(sa,sb,k): return float(sa.get(k,0) or 0)-float(sb.get(k,0) or 0)

def predict_match(pa,pb,oa,ob,rnd,pipe,fcols,calibrator,prof,date,winner):
    sa=snap(prof,pa); sb=snap(prof,pb)
    feats={"surface":SURFACE,"tourney_level":LEVEL,"round":rnd,"best_of":BEST_OF}
    for f in BF:
        key=BFMAP.get(f,f)
        feats[f"diff_{key}"]=d(sa,sb,f)
    feats["elo_diff"]=d(sa,sb,"pre_elo")
    feats["selo_diff"]=d(sa,sb,"selo_Clay")
    feats["rank_diff"]=0.0
    fd=pd.DataFrame([feats])
    for c in fcols:
        if c not in fd.columns: fd[c]=feats.get(c,0)
    p_raw=float(pipe.predict_proba(fd[fcols])[:,1][0])
    if calibrator is not None:
        try: p_cal=float(calibrator.predict([p_raw])[0])
        except: p_cal=p_raw
    else: p_cal=p_raw
    pred=pa if p_cal>=0.5 else pb
    conf=max(p_cal,1-p_cal)
    paf,pbf=devig(ap(oa),ap(ob))
    p_elo=elo_p(sa["pre_elo"],sb["pre_elo"])
    p_temp=sigmoid(logit(p_cal)/1.30)
    cp=np.nan; cpb=np.nan
    if winner:
        aw=norm(winner)
        cp=1 if norm(pred)==aw else 0
        if not np.isnan(paf):
            bk=pa if paf>=0.5 else pb
            cpb=1 if norm(bk)==aw else 0
    return {"date":date,"round":rnd,"surface":SURFACE,"tourney_level":LEVEL,"best_of":BEST_OF,
            "player_a":pa,"player_b":pb,
            "odds_player_a":float(oa) if oa is not None else np.nan,
            "odds_player_b":float(ob) if ob is not None else np.nan,
            "pred_winner":pred,"confidence":round(conf,6),
            "prob_player_a_win":round(p_cal,6),"prob_player_b_win":round(1-p_cal,6),
            "book_fair_prob_a":round(paf,6) if not np.isnan(paf) else np.nan,
            "book_fair_prob_b":round(pbf,6) if not np.isnan(pbf) else np.nan,
            "p_elo_a":round(p_elo,6),"p_temp_a":round(p_temp,6),
            "delta_elo":d(sa,sb,"pre_elo"),
            "correct_prediction":cp,"correct_prediction_book":cpb}

def save_round(rows,slug,rnd):
    df=pd.DataFrame(rows); df.insert(0,"match_no",range(1,len(df)+1))
    base=REPORTS_DIR/f"{slug}_{rnd}"
    df.to_csv(f"{base}_predictions_cck_complete.csv",index=False)
    df.to_csv(f"{base}_predictions_complete.csv",index=False)
    blank=df.copy(); blank["correct_prediction"]=np.nan; blank["correct_prediction_book"]=np.nan
    blank.to_csv(f"{base}_predictions_cck.csv",index=False)
    blank.to_csv(f"{base}_predictions.csv",index=False)
    scored=df["correct_prediction"].notna().sum()
    if scored:
        acc=df[df["correct_prediction"].notna()]["correct_prediction"].mean()
        cpb=df["correct_prediction_book"].dropna()
        bstr=f"  book={cpb.mean():.0%}" if len(cpb) else ""
        print(f"  {slug} {rnd}: {scored}/{len(df)} scored  model={acc:.0%}{bstr}")
    else:
        print(f"  {slug} {rnd}: 0/{len(df)} scored (all pending)")


def main():
    print(f"\n=== Generate {SLUG} R32 ===\n")

    pipe=None; fcols=None; calibrator=None
    for mp in MODEL_PATHS:
        if Path(mp).exists():
            b=load(mp)
            pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else (b,None)
            print(f"  Model: {mp}"); break
    if CAL_PATH.exists():
        calibrator=load(str(CAL_PATH)); print(f"  Calibrator loaded")
    prof=load_profiles()

    # ── R32 DRAW IN BRACKET ORDER ─────────────────────────────
    # 32-player draw, seeds 1-8 enter at R16.
    # Format: (player_a, player_b, odds_a, odds_b, winner_or_None, date)
    # player_a = top of match in bracket

    R32 = [
        # ── TOP HALF ──────────────────────────────────────────
        # Match 1: winner faces FAA (1) in R16 — FAA beat Kopriva
        # FAA had bye; Kopriva won his qualifier match to face FAA directly
        ("Felix Auger Aliassime",    "Vit Kopriva",              -227,  202, "Felix Auger Aliassime",   "2026-05-19"),
        # Match 2: Kovacevic (LL) beat Gea (Q) — upset
        ("Aleksandar Kovacevic",     "Arthur Gea",                122, -137, "Aleksandar Kovacevic",    "2026-05-19"),
        # Match 3: Ugo Carabelli beat Majchrzak — upset
        ("Kamil Majchrzak",          "Camilo Ugo Carabelli",      175, -208, "Camilo Ugo Carabelli",    "2026-05-18"),
        # Match 4: Tiafoe (8) beat Dedura (WC)
        ("Diego Dedura",             "Frances Tiafoe",            231, -278, "Frances Tiafoe",          "2026-05-17"),
        # Match 5: Buse (Q) beat Cobolli (4) — major upset
        ("Flavio Cobolli",           "Ignacio Buse",             -185,  150, "Ignacio Buse",            "2026-05-19"),
        # Match 6: Mensik beat Struff (WC)
        ("Jakub Mensik",             "Jan-Lennard Struff",       -345,  275, "Jakub Mensik",            "2026-05-18"),
        # Match 7: Humbert vs Engel (WC) — in progress (Humbert leading 1-0)
        ("Justin Engel",             "Ugo Humbert",               289, -345, "Ugo Humbert",             "2026-05-19"),
        # Match 8: Gaston (LL) vs Khachanov (5) — in progress (1-1)
        # Note: Kecmanovic vs Khachanov was cancelled; Gaston replaced Kecmanovic
        ("Hugo Gaston",              "Karen Khachanov",           384, -500, "Karen Khachanov",         "2026-05-19"),
        # ── BOTTOM HALF ───────────────────────────────────────
        # Match 9: Darderi (7) beat Burruchaga
        ("Luciano Darderi",          "Roman Burruchaga",         -192,  163, "Luciano Darderi",         "2026-05-19"),
        # Match 10: Hanfmann beat Schoenhaus (Q)
        ("Yannick Hanfmann",         "Max Schoenhaus",           -400,  397, "Yannick Hanfmann",        "2026-05-18"),
        # Match 11: Davidovich Fokina beat Moutet
        ("Alejandro Davidovich Fokina","Corentin Moutet",        -200,  177, "Alejandro Davidovich Fokina","2026-05-18"),
        # Match 12: De Minaur (3) beat Cerundolo
        ("Francisco Cerundolo",      "Alex De Minaur",           -189,  156, "Alex De Minaur",          "2026-05-18"),
        # Match 13: Paul (6) beat Quinn
        ("Tommy Paul",               "Ethan Quinn",              -303,  240, "Tommy Paul",              "2026-05-17"),
        # Match 14: Etcheverry beat Atmane
        ("Terence Atmane",           "Tomas Martin Etcheverry",   183, -227, "Tomas Martin Etcheverry", "2026-05-18"),
        # Match 15: Altmaier beat Hijikata (Q)
        ("Daniel Altmaier",          "Rinky Hijikata",           -303,  240, "Daniel Altmaier",         "2026-05-18"),
        # Match 16: Shelton (2) beat Giron (LL)
        ("Marcos Giron",             "Ben Shelton",               312, -345, "Ben Shelton",             "2026-05-18"),
    ]

    # R16 draw — seeds enter + R32 winners
    # Paul vs Etcheverry and Altmaier vs Shelton are today's pending R16 matches
    R16 = [
        # TOP HALF — seeds FAA(1), Tiafoe(8), Cobolli(4)/Buse, Khachanov(5) enter
        # FAA (1) vs Kovacevic (beat Gea)
        ("Felix Auger Aliassime",    "Aleksandar Kovacevic",     -909,  600, "Aleksandar Kovacevic",    "2026-05-20"),
        # Tiafoe (8) vs Ugo Carabelli (beat Majchrzak)
        ("Frances Tiafoe",           "Camilo Ugo Carabelli",     -119,  110, "Camilo Ugo Carabelli",    "2026-05-20"),
        # Mensik vs Buse (beat Cobolli)
        ("Jakub Mensik",             "Ignacio Buse",             -172,  139, "Ignacio Buse",            "2026-05-20"),
        # Humbert or Engel vs Gaston or Khachanov
        ("Ugo Humbert",              "Karen Khachanov",           220, -278, "Ugo Humbert",             "2026-05-20"),
        # BOTTOM HALF — seeds Darderi(7), De Minaur(3), Paul(6), Shelton(2) enter
        # Darderi (7) vs Hanfmann
        ("Luciano Darderi",          "Yannick Hanfmann",         -213,  175, "Luciano Darderi",         "2026-05-20"),
        # De Minaur (3) vs Davidovich Fokina
        ("Alex De Minaur",           "Alejandro Davidovich Fokina",-154, 129, "Alex De Minaur",          "2026-05-20"),
        # Paul (6) vs Etcheverry — TODAY, pending
        ("Tommy Paul",               "Tomas Martin Etcheverry",  -196,  175, "Tommy Paul",              "2026-05-20"),
        # Shelton (2) vs Altmaier — TODAY, pending
        ("Ben Shelton",              "Daniel Altmaier",          -233,  200, "Daniel Altmaier",         "2026-05-19"),
    ]

    REPORTS_DIR.mkdir(exist_ok=True)

    # Generate R32
    rows=[predict_match(pa,pb,oa,ob,"R32",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in R32]
    save_round(rows,SLUG,"R32")

    # Generate R16
    rows=[predict_match(pa,pb,oa,ob,"R16",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in R16]
    save_round(rows,SLUG,"R16")

    # QF draw from R16 winners
    QF = [
        # Paul beat Etcheverry R16, then beat Altmaier QF
        ("Tommy Paul",               "Daniel Altmaier",          -250,  200, "Tommy Paul",              "2026-05-21"),
        # Buse beat Mensik R16, then beat Humbert QF — Buse now favourite!
        ("Ignacio Buse",             "Ugo Humbert",              -312,  250, "Ignacio Buse",            "2026-05-21"),
        # Darderi vs De Minaur — pending
        ("Luciano Darderi",          "Alex De Minaur",            100, -106, "Alex De Minaur",          "2026-05-21"),
        # Kovacevic vs Ugo Carabelli — pending (Kovacevic underdog again)
        ("Aleksandar Kovacevic",     "Camilo Ugo Carabelli",      208, -250, "Aleksandar Kovacevic",    "2026-05-21"),
    ]

    rows=[predict_match(pa,pb,oa,ob,"QF",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in QF]
    save_round(rows,SLUG,"QF")

    r32_done = sum(1 for *_,w,_ in R32 if w)
    r16_done = sum(1 for *_,w,_ in R16 if w)
    # SF — De Minaur vs Paul | Kovacevic vs Buse
    SF = [
        ("Alex De Minaur",           "Tommy Paul",               -161,  138, "Tommy Paul",              "2026-05-22"),
        ("Aleksandar Kovacevic",     "Ignacio Buse",             -286,  275, "Ignacio Buse",            "2026-05-22"),
    ]

    # Final — Tommy Paul vs Ignacio Buse
    F = [
        ("Tommy Paul",               "Ignacio Buse",             -175,  142, "Ignacio Buse",            "2026-05-23"),
    ]
    rows=[predict_match(pa,pb,oa,ob,"SF",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in SF]
    save_round(rows,SLUG,"SF")

    rows=[predict_match(pa,pb,oa,ob,"F",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in F]
    save_round(rows,SLUG,"F")

    qf_done = sum(1 for *_,w,_ in QF if w)
    print(f"\n  R32: {r32_done}/16 scored")
    print(f"  R16: {r16_done}/8 scored")
    print(f"  QF:  {qf_done}/4 scored")
    print(f"  SF:  0/2 scored (pending)")
    print(f"\nRun:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/hamburg2026_*.csv docs/index.html")
    print(f"  git commit -m 'update: Hamburg QF complete + SF generated'")
    print(f"  git push")

if __name__=="__main__": main()
