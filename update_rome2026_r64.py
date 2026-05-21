"""
update_rome2026_r64.py
=======================
1. Fixes R128 errors (Ofner beat Michelsen, not Michelsen)
2. Fills remaining 4 R128 pending results
3. Generates R64 with ALL 32 matches and odds
4. Scores all completed R64 matches

Notes:
  - Vacherot (14) withdrew -- replaced by Cilic vs Landaluce
  - Sinner vs Ofner odds: -10000/+2000 (essentially certain)

Run from your project root:
    python update_rome2026_r64.py
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
    acc=df[df["correct_prediction"].notna()]["correct_prediction"].mean() if scored else float("nan")
    book=df[df["correct_prediction_book"].notna()]["correct_prediction_book"].mean() if df["correct_prediction_book"].notna().sum() else float("nan")
    if scored:
        print(f"  {slug} {rnd}: {scored}/{len(df)} scored  model={acc:.0%}  book={book:.0%}" if not np.isnan(book) else f"  {slug} {rnd}: {scored}/{len(df)} scored  model={acc:.0%}")
    else:
        print(f"  {slug} {rnd}: 0/{len(df)} scored (pending)")


# -- STEP 1: FIX R128 -----------------------------------------

def fix_r128():
    print("\n--- Fixing/completing R128 ---")
    fixes = [
        # Ofner beat Michelsen -- was scored wrong
        ("Sebastian Ofner",          "Alex Michelsen",            "Sebastian Ofner",         -169,  138),
        # 4 pending results
        ("Andrea Pellegrino",        "Luca Nardi",                "Andrea Pellegrino",       -161,  136),
        ("Lorenzo Sonego",           "Ignacio Buse",              "Ignacio Buse",             138, -149),
        ("Hamad Medjedovic",         "Valentin Royer",            "Hamad Medjedovic",        -303,  256),
        ("Thiago Agustin Tirante",   "Gianluca Cadenasso",        "Thiago Agustin Tirante",  -227,  213),
    ]
    for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
        fpath=REPORTS_DIR/f"rome2026_R128{suffix}"
        if not fpath.exists(): print(f"  NOT FOUND: {fpath.name}"); continue
        df=pd.read_csv(fpath)
        all_names=list(set(df["player_a"].dropna().tolist()+df["player_b"].dropna().tolist()))
        changed=False
        def best(t,candidates):
            tn=norm(t)
            e=[c for c in candidates if norm(c)==tn]
            if e: return e[0]
            s=[c for c in candidates if tn in norm(c) or norm(c) in tn]
            if s: return min(s,key=lambda c:abs(len(norm(c))-len(tn)))
            return None
        for pa_r,pb_r,w_r,oa,ob in fixes:
            pa_m=best(pa_r,all_names); pb_m=best(pb_r,all_names)
            if not pa_m or not pb_m: print(f"  NOT FOUND: {pa_r} vs {pb_r}"); continue
            aw=best(w_r,[pa_m,pb_m])
            if not aw: continue
            mask=((df["player_a"]==pa_m)&(df["player_b"]==pb_m))|((df["player_a"]==pb_m)&(df["player_b"]==pa_m))
            if not mask.any(): print(f"  ROW NOT FOUND: {pa_m} vs {pb_m}"); continue
            idx=df[mask].index[0]
            pred=str(df.at[idx,"pred_winner"])
            cp=1 if norm(pred)==norm(aw) else 0
            old=df.at[idx,"correct_prediction"]
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
            flag=" (CORRECTED)" if pd.notna(old) and int(old)!=cp else ""
            print(f"  {pa_m} vs {pb_m} -> {aw} ({'OK' if cp else 'X'}){flag}")
            changed=True
        if changed: df.to_csv(fpath,index=False)

    fpath=REPORTS_DIR/"rome2026_R128_predictions_cck_complete.csv"
    if fpath.exists():
        df=pd.read_csv(fpath)
        cp=pd.to_numeric(df["correct_prediction"],errors="coerce")
        print(f"\n  R128 final: {cp.notna().sum()}/{len(df)} scored  model={cp.mean():.0%}")


# -- STEP 2: GENERATE R64 -------------------------------------

def generate_r64(pipe,fcols,calibrator,prof):
    print("\n--- Generating R64 (all 32 matches) ---")

    # Format: (player_a, player_b, odds_a, odds_b, winner_or_None, date)
    # Bracket order top to bottom. player_a = top of match.
    # Completed matches (May 8) scored. May 9 matches pending.
    # NOTE: Vacherot (14) withdrew -> replaced by Cilic vs Landaluce

    R64 = [
        # -- TOP HALF -----------------------------------------
        # Match 1: Sinner (1) vs Ofner -- May 9, pending
        ("Jannik Sinner",             "Sebastian Ofner",          -10000, 2000, None,                      "2026-05-09"),
        # Match 2: Mensik (26) vs Popyrin -- May 9, pending
        ("Jakub Mensik",              "Alexei Popyrin",             -244,  220, None,                      "2026-05-09"),
        # Match 3: Tiafoe (20) vs Buse -- May 9, pending
        ("Frances Tiafoe",            "Ignacio Buse",               -167,  139, None,                      "2026-05-09"),
        # Match 4: Fils (15) vs Pellegrino -- May 9, pending
        ("Arthur Fils",               "Andrea Pellegrino",         -1111,  740, None,                      "2026-05-09"),
        # Match 5: Rublev (12) vs Kecmanovic -- May 9, pending
        ("Andrey Rublev",             "Miomir Kecmanovic",          -250,  220, None,                      "2026-05-09"),
        # Match 6: Davidovich Fokina (21) vs Garin -- May 9, pending
        ("Alejandro Davidovich Fokina","Cristian Garin",            -161,  139, None,                      "2026-05-09"),
        # Match 7: Nakashima (30) vs Bautista Agut -- May 9, pending
        ("Brandon Nakashima",         "Roberto Bautista Agut",      -189,  163, None,                      "2026-05-09"),
        # Match 8: Shelton (5) vs Basilashvili -- May 9, pending
        ("Ben Shelton",               "Nikoloz Basilashvili",       -370,  300, None,                      "2026-05-09"),
        # Match 9: FAA (4) vs Navone -- May 9, pending
        ("Felix Auger Aliassime",     "Mariano Navone",             -185,  150, None,                      "2026-05-09"),
        # Match 10: Fonseca (27) vs Medjedovic -- May 9, pending
        ("Joao Fonseca",              "Hamad Medjedovic",           -179,  163, None,                      "2026-05-09"),
        # Match 11: Etcheverry (24) vs Bellucci -- May 9, pending
        ("Tomas Martin Etcheverry",   "Mattia Bellucci",            -294,  250, None,                      "2026-05-09"),
        # Match 12: Vacherot (14) withdrew -> Cilic vs Landaluce -- May 9, pending
        ("Marin Cilic",               "Martin Landaluce",           -175,  150, None,                      "2026-05-09"),
        # Match 13: Cobolli (10) vs Atmane -- May 9, pending
        ("Flavio Cobolli",            "Terence Atmane",             -270,  240, None,                      "2026-05-09"),
        # Match 14: Norrie (17) vs Tirante -- May 9, pending
        ("Cameron Norrie",            "Thiago Agustin Tirante",     -167,  138, None,                      "2026-05-09"),
        # Match 15: Moutet (28) vs Llamas Ruiz -- May 9, pending
        ("Corentin Moutet",           "Pablo Llamas Ruiz",          -137,  125, None,                      "2026-05-09"),
        # Match 16: Medvedev (7) vs Machac -- May 9, pending
        ("Daniil Medvedev",           "Tomas Machac",               -159,  129, None,                      "2026-05-09"),
        # -- BOTTOM HALF --------------------------------------
        # Match 17: Musetti (8) vs Mpetshi Perricard -- COMPLETED May 8
        ("Lorenzo Musetti",           "Giovanni Mpetshi Perricard", -833,  700, "Lorenzo Musetti",         "2026-05-08"),
        # Match 18: Cerundolo F (25) vs Tabilo -- COMPLETED May 8
        ("Francisco Cerundolo",       "Alejandro Tabilo",           -149,  129, "Francisco Cerundolo",     "2026-05-08"),
        # Match 19: Ruud (23) vs Svajda -- COMPLETED May 8
        ("Casper Ruud",               "Zachary Svajda",            -1667,  900, "Casper Ruud",             "2026-05-08"),
        # Match 20: Lehecka (11) vs Struff -- COMPLETED May 8 -- upset
        ("Jiri Lehecka",              "Jan-Lennard Struff",         -385,  300, "Jiri Lehecka",            "2026-05-08"),
        # Match 21: Khachanov (13) vs Shevchenko -- COMPLETED May 8
        ("Karen Khachanov",           "Alexander Shevchenko",       -250,  216, "Karen Khachanov",         "2026-05-08"),
        # Match 22: Humbert (31) vs Kopriva -- COMPLETED May 8 -- upset (Kopriva favoured)
        ("Ugo Humbert",               "Vit Kopriva",                 199, -227, "Ugo Humbert",             "2026-05-08"),
        # Match 23: Kovacevic LL vs VdZ -- COMPLETED May 8
        ("Aleksandar Kovacevic",      "Botic van de Zandschulp",    -400,  309, "Botic van de Zandschulp", "2026-05-08"),
        # Match 24: Djokovic (3) vs Prizmic -- COMPLETED May 8 -- MAJOR UPSET
        ("Novak Djokovic",            "Dino Prizmic",               -227,  190, "Dino Prizmic",            "2026-05-08"),
        # Match 25: De Minaur (6) vs Arnaldi -- COMPLETED May 8 -- upset
        ("Alex De Minaur",            "Matteo Arnaldi",             -233,  220, "Matteo Arnaldi",          "2026-05-08"),
        # Match 26: Jodar (32) vs Borges -- COMPLETED May 8 -- upset
        ("Rafael Jodar",              "Nuno Borges",                -455,  363, "Rafael Jodar",            "2026-05-08"),
        # Match 27: Tien (19) vs Dzumhur -- COMPLETED May 8
        ("Learner Tien",              "Damir Dzumhur",              -123,  110, "Learner Tien",            "2026-05-08"),
        # Match 28: Bublik (9) vs Baez -- COMPLETED May 8
        ("Alexander Bublik",          "Sebastian Baez",             -152,  125, "Alexander Bublik",        "2026-05-08"),
        # Match 29: Paul (16) vs Vukic -- COMPLETED May 8
        ("Tommy Paul",                "Aleksandar Vukic",          -1250,  800, "Tommy Paul",              "2026-05-08"),
        # Match 30: Darderi (18) vs Hanfmann -- COMPLETED May 8 -- Darderi favoured
        ("Luciano Darderi",           "Yannick Hanfmann",           -182,  175, "Luciano Darderi",         "2026-05-08"),
        # Match 31: Griekspoor (29) vs Blockx -- COMPLETED May 8 -- upset
        ("Tallon Griekspoor",         "Alexander Blockx",           -208,  175, "Alexander Blockx",        "2026-05-08"),
        # Match 32: Zverev (2) vs Altmaier -- COMPLETED May 8
        ("Alexander Zverev",          "Daniel Altmaier",           -1429,  800, "Alexander Zverev",        "2026-05-08"),
    ]

    REPORTS_DIR.mkdir(exist_ok=True)
    rows=[predict_match(pa,pb,oa,ob,"R64",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in R64]
    save_round(rows,SLUG,"R64")
    completed=sum(1 for *_,w,_ in R64 if w)
    pending=sum(1 for *_,w,_ in R64 if not w)
    print(f"  {completed} scored, {pending} pending (May 9 matches)")


def main():
    print("\n=== Update Rome 2026: R128 fixes + Full R64 ===\n")
    pipe=None; fcols=None; calibrator=None
    for mp in MODEL_PATHS:
        if Path(mp).exists():
            b=load(mp)
            pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else (b,None)
            print(f"  Model: {mp}"); break
    if CAL_PATH.exists():
        calibrator=load(str(CAL_PATH)); print(f"  Calibrator loaded")
    prof=load_profiles()

    fix_r128()
    generate_r64(pipe,fcols,calibrator,prof)

    print(f"\nDone. Run:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/rome2026_*.csv docs/index.html")
    print(f"  git commit -m 'update: Rome R128 complete + R64 generated (16/32 scored)'")
    print(f"  git push")

if __name__=="__main__": main()
