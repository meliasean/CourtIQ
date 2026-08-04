#!/usr/bin/env python3
"""
Fix the Form column on the players page.

ROOT CAUSE
----------
renderPlayers() reads p.last5 and JSON.parses it into W/L badges. Nothing in
courtiq_engine.py ever WRITES last5 - it appears exactly once in the file, in
the line that reads it. update_profiles() writes form10_wr / form5_wr / streak
but not last5, so once that function wrote the profile CSV the column was gone.
JSON.parse(undefined) throws, the bare catch(e){} swallows it, and the cell
keeps its em-dash. Streak still renders because streak IS written.

THREE EDITS
-----------
 A  renderer: fall back to form5_wr as a percentage when last5 is absent,
    so the column shows real data even on profiles written before the fix.
 B  update_profiles, existing-player branch: write last5, merging any prior
    last5 with this event's results so history survives across tournaments.
 C  update_profiles, new-player branch: write last5 from the event.

Idempotent - safe to re-run. DRY RUN by default.

Usage:
    python apply_form_fix.py                       # preview
    python apply_form_fix.py --commit              # write (makes .bak)
    python apply_form_fix.py --file path.py --commit
"""
import sys, os, shutil
from datetime import datetime

NEW_RENDER = (
    """    let f='<span style="color:var(--txt2)">&#8212;</span>';"""
    """try{{const l5=p.last5?JSON.parse(p.last5):[];"""
    """if(l5.length>0){{f=l5.slice(-5).map(m=>{{"""
    """const c=m.result==='W'?'rgba(74,222,128,0.15)':'rgba(239,68,68,0.15)';"""
    """const t=m.result==='W'?'#4ade80':'#ef4444';"""
    """return '<span style="font-family:var(--mono);font-size:10px;padding:1px 5px;"""
    """border-radius:3px;margin-left:2px;font-weight:600;background:'+c+';color:'+t+'">'"""
    """+m.result+'</span>';}}).join('');}}"""
    """else if(p.form5_wr!=null&&p.form5_wr!==''&&!isNaN(p.form5_wr)){{"""
    """const pct=Math.round(Number(p.form5_wr)*100);"""
    """const t=pct>=60?'#4ade80':pct<=40?'#ef4444':'var(--txt2)';"""
    """f='<span style="font-family:var(--mono);font-size:11px;color:'+t+'" """
    """title="last-5 win rate">'+pct+'%</span>';}}"""
    """}}catch(e){{}}"""
)

BLOCK_B = '''            # Last-5 results (JSON) for the players-page Form column.
            # Merge prior last5 with this event so history survives rounds.
            try:
                _prior = json.loads(str(seed_row.iloc[-1].get("last5") or "[]"))
                if not isinstance(_prior, list):
                    _prior = []
            except Exception:
                _prior = []
            _event = [
                {"result": "W" if int(r["is_win"]) == 1 else "L",
                 "date": r["date"].strftime("%Y-%m-%d")}
                for _, r in sub.sort_values("date").iterrows()
            ]
            updated.loc[idx, "last5"] = json.dumps((_prior + _event)[-5:])
'''

LINE_C = '''                "last5": json.dumps([
                    {"result": "W" if int(r["is_win"]) == 1 else "L",
                     "date": r["date"].strftime("%Y-%m-%d")}
                    for _, r in sub.sort_values("date").iterrows()
                ][-5:]),
'''


def main():
    commit = "--commit" in sys.argv
    path = "courtiq_engine.py"
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]

    if not os.path.exists(path):
        print(f"ABORT: {path} not found"); sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    applied, skipped = [], []

    # ---- Edit A: renderer -------------------------------------------------
    idx_a = [i for i, L in enumerate(lines) if "const l5=p.last5" in L]
    if not idx_a:
        print("ABORT: renderer line (const l5=p.last5) not found"); sys.exit(1)
    if len(idx_a) > 1:
        print(f"ABORT: {len(idx_a)} renderer lines matched; expected 1"); sys.exit(1)
    i = idx_a[0]
    if "form5_wr" in lines[i]:
        skipped.append(f"A renderer (line {i+1}) - fallback already present")
    else:
        lines[i] = NEW_RENDER
        applied.append(f"A renderer (line {i+1}) - added form5_wr fallback")

    # ---- Edit B: existing-player branch ----------------------------------
    idx_b = [i for i, L in enumerate(lines)
             if L.strip() == 'updated.loc[idx, "streak"] = float(streak)']
    if not idx_b:
        print("ABORT: anchor for edit B not found "
              "(updated.loc[idx, \"streak\"] = float(streak))"); sys.exit(1)
    if len(idx_b) > 1:
        print(f"ABORT: {len(idx_b)} anchors matched for edit B; expected 1"); sys.exit(1)
    i = idx_b[0]
    if any('updated.loc[idx, "last5"]' in L for L in lines):
        skipped.append("B update_profiles existing-player - last5 already written")
    else:
        lines[i:i+1] = [lines[i]] + BLOCK_B.rstrip("\n").split("\n")
        applied.append(f"B update_profiles existing-player (after line {i+1}) - writes last5")

    # ---- Edit C: new-player branch ---------------------------------------
    idx_c = [i for i, L in enumerate(lines) if L.strip() == '"streak": float(streak),']
    if not idx_c:
        print("ABORT: anchor for edit C not found (\"streak\": float(streak),)"); sys.exit(1)
    if len(idx_c) > 1:
        print(f"ABORT: {len(idx_c)} anchors matched for edit C; expected 1"); sys.exit(1)
    i = idx_c[0]
    if any('"last5": json.dumps([' in L for L in lines):
        skipped.append("C update_profiles new-player - last5 already written")
    else:
        lines[i:i+1] = [lines[i]] + LINE_C.rstrip("\n").split("\n")
        applied.append(f"C update_profiles new-player (after line {i+1}) - writes last5")

    out = nl.join(lines)

    # syntax gate - never write a file that will not import
    import ast
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    print("Edits:")
    for a in applied: print(f"  APPLY  {a}")
    for s in skipped: print(f"  SKIP   {s}")
    print("\nSyntax check on patched source: OK")

    if not applied:
        print("\nNothing to do - already patched.")
        return
    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}   (backup: {bak})")
    print("\nNEXT:")
    print("  python courtiq_engine.py site --output docs/index.html")
    print("  # Form shows percentages now. W/L badges return for any player")
    print("  # whose profile is rewritten after this patch, i.e. after the")
    print("  # post-Canada rebuild.")


if __name__ == "__main__":
    main()
