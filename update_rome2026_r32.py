"""
update_rome2026_r32.py
=======================
1. Completes R64 scoring (all 32 matches verified)
2. Generates R32 with all 16 matches
3. Scores all completed R32 matches

VERIFIED CORRECTIONS:
  - Walkover: Machac T. listed first = Machac RECEIVES walkover (Medvedev withdrew)
    -> Machac advances to R32, faces Llamas Ruiz
  - Fils RETIRED: Pellegrino listed first, Fils second, Fils was -1250 favourite
    -> Fils retired, Pellegrino advances
  - R32 May 11 results now in (4 upsets):
    Pellegrino beat Tiafoe, Basilashvili beat Nakashima,
    Landaluce beat Bellucci, Rublev beat Davidovich Fokina

Run:
    python update_rome2026_r32.py
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
        if oa is not None:
            df.at[idx,"odds_player_a"]=float(oa)
            df.at[idx,"odds_player_b"]=float(ob)
        try:
            oa_v=pd.to_numeric(df.at[idx,"odds_player_a"],errors="coerce")
            ob_v=pd.to_numeric(df.at[idx,"odds_player_b"],errors="coerce")
            paf,_=devig(ap(oa_v),ap(ob_v))
            if not np.isnan(paf):
                bk=df.at[idx,"player_a"] if paf>=0.5 else df.at[idx,"player_b"]
                df.at[idx,"correct_prediction_book"]=1 if norm(bk)==norm(aw) else 0
                df.at[idx,"book_fair_prob_a"]=round(float(paf),6)
        except: pass
        print(f"  {pa_m} vs {pb_m} -> {aw} ({'OK' if cp else 'X'})")
        changed=True
    if changed: df.to_csv(fpath,index=False)


# -- ALL VERIFIED R64 RESULTS ---------------------------------

R64_RESULTS = [
    # -- May 8 (bottom half) ------------------------------------
    ("Yannick Hanfmann",           "Luciano Darderi",            "Luciano Darderi",           -182,  175),
    ("Lorenzo Musetti",            "Giovanni Mpetshi Perricard", "Lorenzo Musetti",            -833,  700),
    ("Ugo Humbert",                "Vit Kopriva",                "Ugo Humbert",                -227,  199),
    ("Sebastian Baez",             "Alexander Bublik",           "Alexander Bublik",           -152,  125),
    ("Novak Djokovic",             "Dino Prizmic",               "Dino Prizmic",               -227,  190),
    ("Tommy Paul",                 "Aleksandar Vukic",           "Tommy Paul",                -1250,  800),
    ("Tallon Griekspoor",          "Alexander Blockx",           "Alexander Blockx",           -208,  175),
    ("Learner Tien",               "Damir Dzumhur",              "Learner Tien",               -123,  110),
    ("Francisco Cerundolo",        "Alejandro Tabilo",           "Francisco Cerundolo",        -149,  129),
    ("Alexander Zverev",           "Daniel Altmaier",            "Alexander Zverev",          -1429,  800),
    ("Casper Ruud",                "Zachary Svajda",             "Casper Ruud",               -1667,  900),
    ("Botic van de Zandschulp",    "Aleksandar Kovacevic",       "Botic van de Zandschulp",    -400,  309),
    ("Jiri Lehecka",               "Jan-Lennard Struff",         "Jiri Lehecka",               -385,  300),
    ("Matteo Arnaldi",             "Alex De Minaur",             "Matteo Arnaldi",             -233,  220),
    ("Nuno Borges",                "Rafael Jodar",               "Rafael Jodar",               -455,  363),
    ("Karen Khachanov",            "Alexander Shevchenko",       "Karen Khachanov",            -250,  216),
    # -- May 9 (top half) --------------------------------------
    ("Jakub Mensik",               "Alexei Popyrin",             "Alexei Popyrin",             -244,  200),
    # Fils RETIRED -> Pellegrino advances
    ("Arthur Fils",                "Andrea Pellegrino",          "Andrea Pellegrino",         -1250,  800),
    ("Joao Fonseca",               "Hamad Medjedovic",           "Hamad Medjedovic",           -192,  175),
    ("Jannik Sinner",              "Sebastian Ofner",            "Jannik Sinner",            -10000, 2500),
    ("Frances Tiafoe",             "Ignacio Buse",               "Frances Tiafoe",             -133,  129),
    ("Nikoloz Basilashvili",       "Ben Shelton",                "Nikoloz Basilashvili",       -400,  333),
    ("Flavio Cobolli",             "Terence Atmane",             "Flavio Cobolli",             -278,  231),
    ("Brandon Nakashima",          "Roberto Bautista Agut",      "Brandon Nakashima",          -200,  165),
    ("Tomas Martin Etcheverry",    "Mattia Bellucci",            "Mattia Bellucci",            -333,  275),
    ("Alejandro Davidovich Fokina","Cristian Garin",             "Alejandro Davidovich Fokina",-141,  129),
    ("Felix Auger Aliassime",      "Mariano Navone",             "Mariano Navone",             -189,  164),
    ("Marin Cilic",                "Martin Landaluce",           "Martin Landaluce",           -175,  150),
    ("Andrey Rublev",              "Miomir Kecmanovic",          "Andrey Rublev",              -250,  208),
    ("Corentin Moutet",            "Pablo Llamas Ruiz",          "Pablo Llamas Ruiz",          -143,  120),
    ("Cameron Norrie",             "Thiago Agustin Tirante",     "Thiago Agustin Tirante",     -167,  129),
    # WALKOVER: Machac listed first = Machac receives walkover (Medvedev withdrew)
    ("Tomas Machac",               "Daniil Medvedev",            "Daniil Medvedev",             -147,  129),  # Machac withdrew, Medvedev receives walkover
]


# -- ALL VERIFIED R32 MATCHES ---------------------------------
# Format: (player_a, player_b, odds_a, odds_b, winner_or_None, date)

R32_DRAW = [
    # -- TOP HALF -- May 11 -------------------------------------
    # Sinner (1) vs Popyrin (beat Mensik)
    ("Jannik Sinner",              "Alexei Popyrin",            -5000, 1800, "Jannik Sinner",            "2026-05-11"),
    # Tiafoe (20) vs Pellegrino (Fils retired) -- PELLEGRINO WON upset
    ("Frances Tiafoe",             "Andrea Pellegrino",          -244,  200, "Andrea Pellegrino",        "2026-05-11"),
    # Rublev (12) vs Davidovich Fokina (21) -- RUBLEV WON
    ("Andrey Rublev",              "Alejandro Davidovich Fokina",-161,  144, "Andrey Rublev",            "2026-05-11"),
    # Nakashima (30) vs Basilashvili (beat Shelton) -- BASILASHVILI WON upset
    ("Brandon Nakashima",          "Nikoloz Basilashvili",       -154,  129, "Nikoloz Basilashvili",     "2026-05-11"),
    # Navone (beat FAA) vs Medjedovic (beat Fonseca) -- pending
    ("Mariano Navone",             "Hamad Medjedovic",            121, -137, "Hamad Medjedovic",         "2026-05-11"),
    # Bellucci (beat Etcheverry) vs Landaluce (beat Cilic) -- LANDALUCE WON upset
    ("Mattia Bellucci",            "Martin Landaluce",           -115,  100, "Martin Landaluce",         "2026-05-11"),
    # Cobolli (10) vs Tirante (beat Norrie) -- pending
    ("Flavio Cobolli",             "Thiago Agustin Tirante",     -256,  220, "Thiago Agustin Tirante",   "2026-05-11"),
    # Llamas Ruiz (beat Moutet) vs Machac (received walkover -- Medvedev withdrew)
    ("Pablo Llamas Ruiz",          "Daniil Medvedev",             175, -217, "Daniil Medvedev",          "2026-05-11"),
    # -- BOTTOM HALF -- May 10 completed -----------------------
    # Ruud (23) beat Lehecka (11)
    ("Casper Ruud",                "Jiri Lehecka",               -196,  163, "Casper Ruud",              "2026-05-10"),
    # Zverev (2) beat Blockx
    ("Alexander Zverev",           "Alexander Blockx",           -476,  351, "Alexander Zverev",         "2026-05-10"),
    # Prizmic beat Humbert -- upset, Prizmic continues incredible run
    ("Ugo Humbert",                "Dino Prizmic",               -278,  225, "Dino Prizmic",             "2026-05-10"),
    # Khachanov (13) beat VdZ
    ("Karen Khachanov",            "Botic van de Zandschulp",    -149,  121, "Karen Khachanov",          "2026-05-10"),
    # Musetti (8) beat Cerundolo F (25)
    ("Lorenzo Musetti",            "Francisco Cerundolo",        -118,  102, "Lorenzo Musetti",          "2026-05-10"),
    # Tien (19) beat Bublik (9)
    ("Learner Tien",               "Alexander Bublik",           -217,  175, "Learner Tien",             "2026-05-10"),
    # Darderi (18) beat Paul (16) -- upset
    ("Tommy Paul",                 "Luciano Darderi",            -137,  119, "Luciano Darderi",          "2026-05-10"),
    # Jodar (32) beat Arnaldi -- Jodar's incredible run continues
    ("Matteo Arnaldi",             "Rafael Jodar",               -345,  275, "Rafael Jodar",             "2026-05-10"),
]


def main():
    print("\n=== Update Rome 2026: Complete R64 + Full R32 ===\n")

    pipe=None; fcols=None; calibrator=None
    for mp in MODEL_PATHS:
        if Path(mp).exists():
            b=load(mp)
            pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else (b,None)
            print(f"  Model: {mp}"); break
    if CAL_PATH.exists():
        calibrator=load(str(CAL_PATH)); print(f"  Calibrator loaded")
    prof=load_profiles()

    # -- Step 1: Complete R64 ---------------------------------
    print("\n--- Completing R64 ---")
    for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
        apply_results(REPORTS_DIR/f"rome2026_R64{suffix}", R64_RESULTS)

    fpath=REPORTS_DIR/"rome2026_R64_predictions_cck_complete.csv"
    if fpath.exists():
        df=pd.read_csv(fpath)
        cp=pd.to_numeric(df["correct_prediction"],errors="coerce")
        cpb=pd.to_numeric(df.get("correct_prediction_book",""),errors="coerce")
        print(f"\n  R64 final: {cp.notna().sum()}/{len(df)} scored  "
              f"model={cp.mean():.0%}  book={cpb.mean():.0%}")

    # -- Step 2: Generate R32 ---------------------------------
    print("\n--- Generating R32 ---")
    rows=[predict_match(pa,pb,oa,ob,"R32",pipe,fcols,calibrator,prof,date,winner)
          for pa,pb,oa,ob,winner,date in R32_DRAW]
    save_round(rows,SLUG,"R32")

    scored=sum(1 for *_,w,_ in R32_DRAW if w)
    pending=sum(1 for *_,w,_ in R32_DRAW if not w)
    print(f"  {scored} scored, {pending} pending")

    print(f"\nDone. Run:")
    print(f"  python courtiq_engine.py site --output docs/index.html")
    print(f"  git add reports/rome2026_*.csv docs/index.html")
    print(f"  git commit -m 'update: Rome R64 complete + R32 (12/16 scored)'")
    print(f"  git push")

if __name__=="__main__": main()
