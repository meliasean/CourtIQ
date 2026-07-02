#!/usr/bin/env python3
"""
Score Wimbledon 2026 R128. 58 of 64 results known; 6 pending suspended matches.
Uncomment PENDING entries and rerun when those finish.
Usage:  python score_wimbledon_r128.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Jannik Sinner", "Miomir Kecmanovic"): "Jannik Sinner",
    ("Nuno Borges", "Tristan Boyer"): "Nuno Borges",
    ("Aleksandar Vukic", "Jenson Brooksby"): "Jenson Brooksby",
    ("Emilio Nava", "Ignacio Buse"): "Ignacio Buse",
    ("Rafael Jodar", "Felix Gill"): "Rafael Jodar",
    ("Denis Shapovalov", "Pablo Carreno Busta"): "Pablo Carreno Busta",
    ("Shintaro Mochizuki", "Max Basing"): "Shintaro Mochizuki",
    ("Ethan Quinn", "Luciano Darderi"): "Ethan Quinn",
    ("Casper Ruud", "Hubert Hurkacz"): "Hubert Hurkacz",
    ("Hamad Medjedovic", "Sebastian Ofner"): "Sebastian Ofner",
    ("Soonwoo Kwon", "Martin Landaluce"): "Soonwoo Kwon",
    ("Alexandre Muller", "Tommy Paul"): "Tommy Paul",
    ("Brandon Nakashima", "Jack Pinnington Jones"): "Brandon Nakashima",
    ("Jan-Lennard Struff", "Sebastian Baez"): "Jan-Lennard Struff",
    ("Camilo Ugo Carabelli", "Daniel Merida"): "Daniel Merida",
    ("Marin Cilic", "Daniil Medvedev"): "Daniil Medvedev",
    ("Felix Auger-Aliassime", "Aleksandr Shevchenko"): "Felix Auger-Aliassime",
    ("Adam Walton", "Dino Prizmic"): "Dino Prizmic",
    ("Adolfo Daniel Vallejo", "Nicolas Mejia"): "Nicolas Mejia",
    ("Michael Zheng", "Cameron Norrie"): "Michael Zheng",
    ("Alejandro Davidovich Fokina", "Juan Manuel Cerundolo"): "Alejandro Davidovich Fokina",
    ("Thiago Agustin Tirante", "Fabian Marozsan"): "Fabian Marozsan",
    ("Luca Van Assche", "Marton Fucsovics"): "Marton Fucsovics",
    ("Dalibor Svrcina", "Learner Tien"): "Learner Tien",
    ("Andrey Rublev", "Roman Safiullin"): "Roman Safiullin",
    ("Aleksandar Kovacevic", "Botic van de Zandschulp"): "Botic van de Zandschulp",
    ("Jesper de Jong", "Rinky Hijikata"): "Jesper de Jong",
    ("Roberto Bautista Agut", "Joao Fonseca"): "Joao Fonseca",
    ("Arthur Rinderknech", "Oliver Tarvet"): "Arthur Rinderknech",
    ("Marco Trungelliti", "Martin Damm"): "Martin Damm",
    ("Hugo Gaston", "Stefanos Tsitsipas"): "Stefanos Tsitsipas",
    ("Yibing Wu", "Novak Djokovic"): "Novak Djokovic",
    ("Alex De Minaur", "Roman Andres Burruchaga"): "Alex De Minaur",
    ("Adrian Mannarino", "Titouan Droguet"): "Adrian Mannarino",
    ("Pablo Llamas Ruiz", "Zachary Svajda"): "Zachary Svajda",
    ("Kamil Majchrzak", "Alejandro Tabilo"): "Kamil Majchrzak",
    ("Karen Khachanov", "Billy Harris"): "Karen Khachanov",
    ("Yannick Hanfmann", "Giovanni Mpetshi Perricard"): "Yannick Hanfmann",
    ("Jakub Mensik", "Toby Samuel"): "Jakub Mensik",
    ("Dane Sweeny", "Grigor Dimitrov"): "Grigor Dimitrov",
    ("Stan Wawrinka", "Matteo Berrettini"): "Matteo Berrettini",
    ("Raphael Collignon", "Arthur Fils"): "Arthur Fils",
    ("Ugo Humbert", "Zizou Bergs"): "Zizou Bergs",
    ("Sho Shimabukuro", "Jaime Faria"): "Jaime Faria",
    ("Damir Dzumhur", "Arthur Fery"): "Arthur Fery",
    ("Otto Virtanen", "Ben Shelton"): "Otto Virtanen",
    ("Taylor Fritz", "Dusan Lajovic"): "Taylor Fritz",
    ("Patrick Kypson", "Mackenzie McDonald"): "Patrick Kypson",
    ("Benjamin Bonzi", "Gabriel Diallo"): "Gabriel Diallo",
    ("Lorenzo Sonego", "Tomas Martin Etcheverry"): "Lorenzo Sonego",
    ("Kyrian Jacquet", "Vilius Gaubas"): "Kyrian Jacquet",
    ("Thanasi Kokkinakis", "Alexander Bublik"): "Alexander Bublik",
    ("Alex Michelsen", "Jacob Fearnley"): "Jacob Fearnley",
    ("Jaume Munar", "Francisco Cerundolo"): "Jaume Munar",
    ("Matteo Arnaldi", "Quentin Halys"): "Quentin Halys",
    ("Corentin Moutet", "Marcos Giron"): "Marcos Giron",
    ("Valentin Royer", "Harry Wendelken"): "Valentin Royer",
    ("Alexander Blockx", "Alexander Zverev"): "Alexander Zverev",
    ("Tallon Griekspoor", "James Duckworth"): "James Duckworth",
    ("Mariano Navone", "Flavio Cobolli"): "Flavio Cobolli",
    ("Frances Tiafoe", "Terence Atmane"): "Frances Tiafoe",
    ("Vit Kopriva", "Jan Choinski"): "Jan Choinski",
    ("Jiri Lehecka", "Alexei Popyrin"): "Jiri Lehecka",
    ("Alex Molcan", "Daniel Altmaier"): "Alex Molcan",
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}

def score_file(path, write_complete_to):
    if not os.path.exists(path):
        print(f"  skip (not found): {path}"); return None
    df = pd.read_csv(path)
    scored = 0
    for i, r in df.iterrows():
        key = frozenset([al(r["player_a"]), al(r["player_b"])])
        if key not in RES: continue
        winner = RES[key]
        if pd.isna(r.get("pred_winner")): continue
        df.at[i, "correct_prediction"] = 1 if al(r["pred_winner"]) == winner else 0
        oa, ob = r.get("odds_player_a"), r.get("odds_player_b")
        if pd.notna(oa) and pd.notna(ob):
            book_pick = al(r["player_a"]) if oa < ob else al(r["player_b"])
            df.at[i, "correct_prediction_book"] = 1 if book_pick == winner else 0
        scored += 1
    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring Wimbledon 2026 R128...")
score_file(os.path.join(reports, "wimbledon2026_R128_predictions.csv"),
           os.path.join(reports, "wimbledon2026_R128_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "wimbledon2026_R128_predictions_cck.csv"),
                 os.path.join(reports, "wimbledon2026_R128_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
print(f"\nTotal results defined: {len(RESULTS)} of 64 (6 pending)")
