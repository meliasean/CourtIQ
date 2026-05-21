"""
update_rome2026_r16.py
=======================
Generates Rome 2026 R16 prediction files and scores completed matches.

R16 draw from R32 winners:
  Top half: Sinner, Pellegrino, Rublev, Basilashvili, Medjedovic, Landaluce, Tirante, Medvedev
  Bottom half: Ruud, Zverev, Prizmic, Khachanov, Musetti, Tien, Darderi, Jodar

Results May 12:
  Khachanov beat Prizmic 2-0 (+163/-192)
  Ruud beat Musetti 2-0 (+163/-196)
  Jodar beat Tien 2-0 (-400/+303)
  Darderi beat Zverev 2-1 (+341/-455) -- major upset
  Sinner beat Pellegrino 2-0 (essentially 1.00/-/+4000)
  Rublev beat Basilashvili 2-1 (-294/+250)
  Medjedovic vs Landaluce -- in progress
  Tirante vs Medvedev -- pending

Run:
    python update_rome2026_r16.py
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
        print(f"  {slug} {rnd}: 0/{len(df)} scored")


# R16 draw in bracket order
# Format: (player_a, player_b, odds_a, odds_b, winner_or_None, date)

R16 = [
    # -- TOP HALF ----------------------------------------------
    # Sinner vs Pellegrino -- COMPLETED, Sinner won easily
    # Sinner was at essentially 1.00 decimal (-? American), +4000 for Pellegrino
    ("Jannik Sinner",             "Andrea Pellegrino",        -10000,  4000, "Jannik Sinner",            "2026-05-12"),
    # Rublev vs Basilashvili -- COMPLETED, Rublev won
    ("Andrey Rublev",             "Nikoloz Basilashvili",      -294,   250, "Andrey Rublev",            "2026-05-12"),
    # Medjedovic vs Landaluce -- in progress (Landaluce leading 1-0 in sets)
    ("Hamad Medjedovic",          "Martin Landaluce",          -185,   150, None,                       "2026-05-12"),
    # Tirante vs Medvedev -- pending
    ("Thiago Agustin Tirante",    "Daniil Medvedev",            131,  -161, None,                       "2026-05-12"),
    # -- BOTTOM HALF -------------------------------------------
    # Khachanov beat Prizmic 2-0 -- Khachanov was underdog (+163)
    ("Karen Khachanov",           "Dino Prizmic",               163,  -192, "Karen Khachanov",          "2026-05-12"),
    # Ruud beat Musetti 2-0 -- Ruud was underdog (+163)
    ("Lorenzo Musetti",           "Casper Ruud",               -196,   163, "Casper Ruud",              "2026-05-12"),
    # Jodar beat Tien 2-0 -- Jodar heavily favoured (-400)
    ("Rafael Jodar",              "Learner Tien",              -400,   303, "Rafael Jodar",             "2026-05-12"),
    # Darderi beat Zverev 2-1 -- huge upset (+341)
    ("Luciano Darderi",           "Alexander Zverev",           341,  -455, "Luciano Darderi",          "2026-05-12"),
]


def main():
    print("\n=== Generate Rome 2026 R16 ===\n")
    pipe=None; fcols=None; calibrator=None
    for mp in MODEL_PATHS:
        if Path(mp).exists():
            b=load(mp)
            pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else (b,None)
            print(f"  Model: {mp}"); break
    if CAL_PATH.exists():
        calibrator=load(str(CAL_PATH)); print(f"  Calibrator loaded")
    prof=load_profiles()

    REPORTS_DIR.mkdir(exist_ok=True)
    rows=[predict_match(pa,pb,oa,ob,"R16",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in R16]
    save_round(rows,SLUG,"R16")

    scored=sum(1 for *_,w,_ in R16 if w)
    pending=sum(1 for *_,w,_ in R16 if not w)
    print(f"  {scored} scored, {pending} pending (Medjedovic/Landaluce in progress, Tirante/Medvedev pending)")

    print(f"\nDone. Run:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/rome2026_R16_*.csv docs/index.html")
    print(f"  git commit -m 'add: Rome R16 (6/8 scored, 2 pending)'")
    print(f"  git push")

if __name__=="__main__": main()
