"""
backtest_v2.py
==============
Regenerates all CourtIQ prediction files using RF Model V2
(with form_divergence feature) and saves to reports_v2/.

Covers Wimbledon 2025 -> Rome 2026. Uses same player profiles as V1
for a fair comparison. Actual results (correct_prediction) are
carried over from V1 files so accuracy can be compared directly.

Run:
    python backtest_v2.py           # generate all files + compare
    python backtest_v2.py --dry-run # preview only
"""

import glob, os, sys, unicodedata, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

warnings.filterwarnings("ignore")

REPORTS_DIR    = Path("./reports")
REPORTS_V2_DIR = Path("./reports_v2")
MODEL_V2_PATH  = Path("./models/rf_model_v2.joblib")
CAL_PATH       = Path("./models/prob_calibrator.joblib")
PROFILE_PATH   = Path("./reports/player_profiles_latest.csv")
DRY_RUN = "--dry-run" in sys.argv

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

def load_model():
    if not MODEL_V2_PATH.exists():
        raise FileNotFoundError(f"V2 model not found: {MODEL_V2_PATH}")
    bundle = load(str(MODEL_V2_PATH))
    pipe, fcols = (bundle["pipeline"], bundle["feature_cols"]) if isinstance(bundle,dict) else (bundle,None)
    calibrator = load(str(CAL_PATH)) if CAL_PATH.exists() else None
    print(f"  Model:      {MODEL_V2_PATH}")
    if calibrator: print(f"  Calibrator: {CAL_PATH}")
    if fcols:
        print(f"  Features:   {len(fcols)}")
        if "diff_form_divergence" in fcols:
            print(f"  V2 feature: diff_form_divergence OK")
        else:
            print(f"  WARNING: diff_form_divergence not in feature cols")
    return pipe, fcols, calibrator

def load_profiles():
    df = pd.read_csv(PROFILE_PATH)
    df["name"] = df["name"].astype(str).str.strip()
    print(f"  Profiles:   {PROFILE_PATH} ({len(df)} players)")
    return df

def snap(prof, player):
    pn = unicodedata.normalize("NFKD", str(player)).encode("ascii","ignore").decode("ascii").strip()
    rows = prof[prof["name"]==pn]
    D = {"pre_elo":1500,"selo_Clay":1500,"avg_rest_days":20,"matches_28d":0,
         "form10_wr":.5,"form5_wr":.5,"streak":0,"overall_wr":.5,"peak_elo":1500,
         "current_elo":1500,"wr_Clay":.5,"wr_Grass":.5,"wr_Hard":.5,
         "h2h_wr_prior":.5,"cnt_Clay":0}
    if rows.empty: return D
    r = rows.iloc[-1]
    def g(c,d=0):
        try: return float(r.get(c,d) or d)
        except: return d
    d10=g("form10_wr",.5); d5=g("form5_wr",.5)
    return {**D,
        "pre_elo":g("current_elo",1500),"current_elo":g("current_elo",1500),
        "peak_elo":g("peak_elo",1500),"selo_Clay":g("selo_Clay",1500),
        "overall_wr":g("overall_wr",.5),"avg_rest_days":g("avg_rest_days",20),
        "matches_28d":g("matches_28d",0),"form10_wr":d10,"form5_wr":d5,
        "streak":g("streak",0),"wr_Clay":g("wr_Clay",.5),"wr_Grass":g("wr_Grass",.5),
        "wr_Hard":g("wr_Hard",.5),"h2h_wr_prior":g("h2h_wr_prior",.5),
        "cnt_Clay":g("cnt_Clay",0),"form_divergence":d10-d5}

BF=["pre_elo","selo_Clay","avg_rest_days","matches_28d","form10_wr","form5_wr",
    "streak","overall_wr","peak_elo","current_elo","wr_Clay","wr_Grass","wr_Hard",
    "h2h_wr_prior","cnt_Clay","form_divergence"]
BFMAP={"selo_Clay":"pre_selo","form10_wr":"rolling_10_winrate",
       "form5_wr":"rolling_5_winrate","overall_wr":"win_rate",
       "form_divergence":"form_divergence"}

def predict_row(row, pipe, fcols, calibrator, prof):
    pa=str(row.get("player_a","")); pb=str(row.get("player_b",""))
    sa=snap(prof,pa); sb=snap(prof,pb)
    surf=str(row.get("surface","Hard")); lvl=str(row.get("tourney_level","M"))
    rnd=str(row.get("round","R32")); bo=int(row.get("best_of",3) or 3)
    oa=pd.to_numeric(row.get("odds_player_a",np.nan),errors="coerce")
    ob=pd.to_numeric(row.get("odds_player_b",np.nan),errors="coerce")

    feats={"surface":surf,"tourney_level":lvl,"round":rnd,"best_of":bo}
    for f in BF:
        key=BFMAP.get(f,f)
        feats[f"diff_{key}"]=float(sa.get(f,0) or 0)-float(sb.get(f,0) or 0)
    feats["elo_diff"]=float(sa.get("pre_elo",1500))-float(sb.get("pre_elo",1500))
    feats["selo_diff"]=float(sa.get("selo_Clay",1500))-float(sb.get("selo_Clay",1500))
    feats["rank_diff"]=0.0

    fd=pd.DataFrame([feats])
    if fcols:
        for c in fcols:
            if c not in fd.columns: fd[c]=feats.get(c,0)
        fd=fd[fcols]

    p_raw=float(pipe.predict_proba(fd)[:,1][0])
    if calibrator is not None:
        try: p_cal=float(calibrator.predict([p_raw])[0])
        except: p_cal=p_raw
    else: p_cal=p_raw

    pred=pa if p_cal>=0.5 else pb
    conf=max(p_cal,1-p_cal)
    paf,pbf=devig(ap(oa),ap(ob))
    p_elo=elo_p(sa["pre_elo"],sb["pre_elo"])
    p_temp=sigmoid(logit(p_cal)/1.30)

    # Reconstruct actual winner -- use book as ground truth (never changes).
    # correct_prediction_book is preserved exactly from the original file.
    # correct_prediction is recomputed using V2's pred_winner vs the actual winner.
    v1_cpb = pd.to_numeric(row.get("correct_prediction_book", np.nan), errors="coerce")
    v1_cp  = pd.to_numeric(row.get("correct_prediction", np.nan), errors="coerce")
    v1_pred = str(row.get("pred_winner",""))
    cp_v2 = np.nan
    cpb_v2 = v1_cpb  # book prediction never changes -- preserve as-is

    if pd.notna(v1_cpb) and not np.isnan(paf):
        # Book favourite is the player with higher fair prob
        fav = pa if paf >= 0.5 else pb
        und = pb if paf >= 0.5 else pa
        # Actual winner: book fav won if cpb==1, underdog won if cpb==0
        actual = fav if int(v1_cpb) == 1 else und
        cp_v2 = 1 if norm(pred) == norm(actual) else 0
    elif pd.notna(v1_cp) and norm(v1_pred):
        # No book data -- fall back to V1 pred + V1 cp
        actual = v1_pred if int(v1_cp)==1 else (pb if norm(v1_pred)==norm(pa) else pa)
        cp_v2 = 1 if norm(pred) == norm(actual) else 0

    return {"date":row.get("date",""),"round":rnd,"surface":surf,
            "tourney_level":lvl,"best_of":bo,"player_a":pa,"player_b":pb,
            "odds_player_a":float(oa) if pd.notna(oa) else np.nan,
            "odds_player_b":float(ob) if pd.notna(ob) else np.nan,
            "pred_winner":pred,"confidence":round(conf,6),
            "prob_player_a_win":round(p_cal,6),"prob_player_b_win":round(1-p_cal,6),
            "book_fair_prob_a":round(paf,6) if not np.isnan(paf) else np.nan,
            "book_fair_prob_b":round(pbf,6) if not np.isnan(pbf) else np.nan,
            "p_elo_a":round(p_elo,6),"p_temp_a":round(p_temp,6),
            "delta_elo":feats["elo_diff"],
            "correct_prediction":cp_v2,"correct_prediction_book":cpb_v2}

def process_and_save(fpath, pipe, fcols, calibrator, prof):
    df=pd.read_csv(fpath)
    if df.empty: return None,None,None

    rows=[]
    for _,row in df.iterrows():
        try: rows.append(predict_row(row,pipe,fcols,calibrator,prof))
        except: rows.append(row.to_dict())

    df_v2=pd.DataFrame(rows)
    if "match_no" not in df_v2.columns:
        df_v2.insert(0,"match_no",range(1,len(df_v2)+1))

    # Save 4 variants to reports_v2/
    base=fpath.name.replace("_predictions_cck_complete.csv","").replace("_predictions_complete.csv","")
    REPORTS_V2_DIR.mkdir(parents=True,exist_ok=True)

    # Complete (with correct_prediction)
    df_v2.to_csv(REPORTS_V2_DIR/f"{base}_predictions_cck_complete.csv",index=False)
    df_v2.to_csv(REPORTS_V2_DIR/f"{base}_predictions_complete.csv",index=False)
    # Blank (without correct_prediction)
    blank=df_v2.copy(); blank["correct_prediction"]=np.nan; blank["correct_prediction_book"]=np.nan
    blank.to_csv(REPORTS_V2_DIR/f"{base}_predictions_cck.csv",index=False)
    blank.to_csv(REPORTS_V2_DIR/f"{base}_predictions.csv",index=False)

    # Return accuracy stats
    cp_v1=pd.to_numeric(df.get("correct_prediction",""),errors="coerce")
    cp_v2=pd.to_numeric(df_v2.get("correct_prediction",""),errors="coerce")
    return cp_v1, cp_v2, len(df_v2)

def main():
    print(f"\n{'='*65}")
    print(f"  CourtIQ V2 Backtest{'  [DRY RUN]' if DRY_RUN else ''}")
    print(f"{'='*65}\n")

    pipe,fcols,calibrator=load_model()
    prof=load_profiles()
    print()

    # Collect input files (CCK + standard-only)
    cck_files=sorted(REPORTS_DIR.glob("*_predictions_cck_complete.csv"))
    cck_files=[f for f in cck_files if "_ALL_" not in f.name and "all_rounds" not in f.name]
    cck_keys={f.name.replace("_predictions_cck_complete.csv","") for f in cck_files}
    std_files=[f for f in sorted(REPORTS_DIR.glob("*_predictions_complete.csv"))
               if "_ALL_" not in f.name and "all_rounds" not in f.name
               and "_cck_" not in f.name
               and f.name.replace("_predictions_complete.csv","") not in cck_keys]
    all_files=cck_files+std_files

    print(f"  Input: {len(cck_files)} CCK + {len(std_files)} standard-only = {len(all_files)} files")
    if DRY_RUN:
        for f in all_files: print(f"    {f.name}")
        print(f"\n  Run without --dry-run to generate reports_v2/")
        return

    tm=0; cv1=0; cv2=0
    tourney_stats={}

    for fpath in all_files:
        slug=fpath.name.split("_predictions")[0]
        tourney=slug.rsplit("_",1)[0]
        cp_v1,cp_v2,n=process_and_save(fpath,pipe,fcols,calibrator,prof)
        if cp_v2 is None: continue
        scored=cp_v2.notna().sum()
        if scored==0: continue
        a1=cp_v1.mean(); a2=cp_v2.mean()
        tm+=scored; cv1+=cp_v1.sum(); cv2+=cp_v2.sum()
        tourney_stats.setdefault(tourney,{"n":0,"v1":0.0,"v2":0.0})
        tourney_stats[tourney]["n"]+=scored
        tourney_stats[tourney]["v1"]+=cp_v1.sum()
        tourney_stats[tourney]["v2"]+=cp_v2.sum()
        diff=(a2-a1)*100
        flag=" ^" if diff>0.5 else (" v" if diff<-0.5 else "")
        print(f"  {slug:<42} V1={a1:.0%}  V2={a2:.0%}  {diff:+.1f}pp{flag}")

    print(f"\n{'-'*65}")
    print(f"  {'Tournament':<28} {'n':>5} {'V1':>7} {'V2':>7} {'?':>7}")
    print(f"  {'-'*55}")
    for t,s in sorted(tourney_stats.items()):
        if not s["n"]: continue
        v1=s["v1"]/s["n"]; v2=s["v2"]/s["n"]
        print(f"  {t:<28} {s['n']:>5} {v1:>6.1%} {v2:>6.1%} {(v2-v1)*100:>+6.1f}pp")

    print(f"\n{'='*65}")
    if tm:
        ov1=cv1/tm; ov2=cv2/tm
        print(f"  OVERALL  {tm} matches  V1={ov1:.1%}  V2={ov2:.1%}  ?={((ov2-ov1)*100):+.2f}pp")
        print(f"  Verdict: {'V2 is better OK' if ov2>ov1 else ('V1 is better' if ov1>ov2 else 'Tied')}")
    print(f"\n  Files written to: {REPORTS_V2_DIR}/")
    print(f"\n  To promote V2 for Roland Garros:")
    print(f"    cp models/rf_model.joblib models/rf_model_v1_backup.joblib")
    print(f"    cp models/rf_model_v2.joblib models/rf_model.joblib")
    print(f"    python fit_calibrator.py")
    print(f"    python rebuild_profiles_new_elo.py")
    print(f"    git add models/ reports/")
    print(f"    git commit -m 'feat: promote RF V2 for Roland Garros 2026'")

if __name__=="__main__": main()
