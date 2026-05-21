"""
update_rome2026_qf.py
======================
1. Fills remaining R16 results (Landaluce beat Medjedovic, Medvedev beat Tirante)
2. Generates QF with all 4 matches and available odds

QF draw:
  Ruud vs Khachanov       (-435/+333) -- May 13, interrupted
  Jodar vs Darderi        (-244/+200) -- May 13, pending
  Landaluce vs Medvedev   (+220/-278) -- May 14
  Sinner vs Rublev        (-3333/+1500) -- May 14

Run:
    python update_rome2026_qf.py
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

def best_match(target, candidates):
    tn=norm(target)
    e=[c for c in candidates if norm(c)==tn]
    if e: return e[0]
    s=[c for c in candidates if tn in norm(c) or norm(c) in tn]
    if s: return min(s,key=lambda c:abs(len(norm(c))-len(tn)))
    return None

def apply_results(fpath, results):
    if not fpath.exists(): print(f"  NOT FOUND: {fpath.name}"); return
    df=pd.read_csv(fpath)
    all_names=list(set(df["player_a"].dropna().tolist()+df["player_b"].dropna().tolist()))
    changed=False
    for pa_r,pb_r,w_r,oa,ob in results:
        pa_m=best_match(pa_r,all_names); pb_m=best_match(pb_r,all_names)
        if not pa_m or not pb_m: print(f"  NOT FOUND: {pa_r} vs {pb_r}"); continue
        aw=best_match(w_r,[pa_m,pb_m])
        if not aw: print(f"  WINNER NOT MATCHED: {w_r}"); continue
        mask=((df["player_a"]==pa_m)&(df["player_b"]==pb_m))|((df["player_a"]==pb_m)&(df["player_b"]==pa_m))
        if not mask.any(): print(f"  ROW NOT FOUND: {pa_m} vs {pb_m}"); continue
        idx=df[mask].index[0]
        if pd.notna(df.at[idx,"correct_prediction"]): continue
        pred=str(df.at[idx,"pred_winner"])
        cp=1 if norm(pred)==norm(aw) else 0
        df.at[idx,"correct_prediction"]=cp
        df.at[idx,"odds_player_a"]=float(oa)
        df.at[idx,"odds_player_b"]=float(ob)
        try:
            paf,_=devig(ap(float(oa)),ap(float(ob)))
            if not np.isnan(paf):
                bk=df.at[idx,"player_a"] if paf>=0.5 else df.at[idx,"player_b"]
                df.at[idx,"correct_prediction_book"]=1 if norm(bk)==norm(aw) else 0
                df.at[idx,"book_fair_prob_a"]=round(float(paf),6)
        except: pass
        print(f"  {pa_m} vs {pb_m} -> {aw} ({'OK' if cp else 'X'})")
        changed=True
    if changed: df.to_csv(fpath,index=False)


# -- R16 REMAINING RESULTS ------------------------------------

R16_RESULTS = [
    # Landaluce beat Medjedovic 2-0 (+150/-185) -- upset
    ("Hamad Medjedovic",   "Martin Landaluce",  "Martin Landaluce",  -185,  150),
    # Medvedev beat Tirante 2-0 (-143/+120)
    ("Thiago Agustin Tirante", "Daniil Medvedev", "Daniil Medvedev",  120, -143),
]


# -- QF DRAW --------------------------------------------------
# Format: (player_a, player_b, odds_a, odds_b, winner_or_None, date)

QF = [
    # Ruud vs Khachanov -- May 13, interrupted (Ruud heavily favoured)
    ("Casper Ruud",        "Karen Khachanov",    -435,  333, None, "2026-05-13"),
    # Jodar vs Darderi -- May 13, pending (Jodar favoured -244)
    ("Rafael Jodar",       "Luciano Darderi",    -244,  200, None, "2026-05-13"),
    # Landaluce vs Medvedev -- May 14 (Medvedev favoured -278)
    ("Martin Landaluce",   "Daniil Medvedev",     220, -278, None, "2026-05-14"),
    # Sinner vs Rublev -- May 14 (Sinner massive favourite -3333)
    ("Jannik Sinner",      "Andrey Rublev",     -3333, 1500, None, "2026-05-14"),
]


def main():
    print("\n=== Update Rome 2026: R16 complete + QF ===\n")

    pipe=None; fcols=None; calibrator=None
    for mp in MODEL_PATHS:
        if Path(mp).exists():
            b=load(mp)
            pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else (b,None)
            print(f"  Model: {mp}"); break
    if CAL_PATH.exists():
        calibrator=load(str(CAL_PATH)); print(f"  Calibrator loaded")
    prof=load_profiles()

    # Step 1: Fill remaining R16 results
    print("\n--- Completing R16 ---")
    for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
        apply_results(REPORTS_DIR/f"rome2026_R16{suffix}", R16_RESULTS)

    fpath=REPORTS_DIR/"rome2026_R16_predictions_cck_complete.csv"
    if fpath.exists():
        df=pd.read_csv(fpath)
        cp=pd.to_numeric(df["correct_prediction"],errors="coerce")
        cpb=pd.to_numeric(df.get("correct_prediction_book",""),errors="coerce")
        print(f"\n  R16 final: {cp.notna().sum()}/{len(df)} scored  "
              f"model={cp.mean():.0%}  book={cpb.mean():.0%}")

    # Step 2: Generate QF
    print("\n--- Generating QF ---")
    REPORTS_DIR.mkdir(exist_ok=True)
    rows=[predict_match(pa,pb,oa,ob,"QF",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in QF]
    save_round(rows,SLUG,"QF")

    pending=sum(1 for *_,w,_ in QF if not w)
    print(f"  0 scored, {pending} pending (all May 13-14)")

    print(f"\nDone. Run:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/rome2026_R16_*.csv reports/rome2026_QF_*.csv docs/index.html")
    print(f"  git commit -m 'update: Rome R16 complete + QF generated'")
    print(f"  git push")

if __name__=="__main__": main()
