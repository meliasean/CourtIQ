#!/usr/bin/env python3
"""
Apply the Console design to courtiq_engine.py's site template.

WHY THIS IS SMALL
-----------------
The template's CSS is token-driven through :root, so most of the visual change
is a token swap. Only four edits are needed:

  A  font link      -> Inter Tight + JetBrains Mono
  B  :root tokens   -> console palette, square corners, new families
  C  .logo-dot      -> square status indicator, no glow
  D  .hdr           -> solid bar, no backdrop blur

Everything else inherits. The data path, the JS, and every render function are
untouched, so `git revert` on this commit restores the previous look exactly.

NOTE ON THE MONO: the mockup used Martian Mono, which is very wide. The
template has 38 var(--mono) usages in layouts metered for DM Mono, so a wide
face would overflow the tabular columns. JetBrains Mono keeps the modern,
sharp character at safe metrics.

Idempotent. DRY RUN by default.

Usage:
    python apply_console_theme.py
    python apply_console_theme.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

MARKER = "--console-theme"

OLD_FONTS = '<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">'
NEW_FONTS = '<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'

OLD_ROOT = """:root{{
  --bg0:#080c10;
  --bg1:#0d1117;
  --bg2:#161b22;
  --bg3:#21262d;
  --bg4:#30363d;
  --line:rgba(255,255,255,0.06);
  --line2:rgba(255,255,255,0.12);
  --txt0:#e6edf3;
  --txt1:#8b949e;
  --txt2:#484f58;
  --green:#3fb950;
  --green-dim:rgba(63,185,80,0.1);
  --green-bright:#56d364;
  --clay:#f0883e;
  --clay-dim:rgba(240,136,62,0.1);
  --blue:#58a6ff;
  --blue-dim:rgba(88,166,255,0.1);
  --purple:#bc8cff;
  --purple-dim:rgba(188,140,255,0.1);
  --gold:#e3b341;
  --mono:'DM Mono',monospace;
  --display:'Syne',sans-serif;
  --sans:'DM Sans',sans-serif;
  --radius:6px;
  --radius-lg:10px;
}}"""

NEW_ROOT = """:root{{
  /* --console-theme: hairline panels, square corners, mono for figures only */
  --bg0:#0d1117;
  --bg1:#11161c;
  --bg2:#161b22;
  --bg3:#1c2229;
  --bg4:#22282f;
  --line:#1c2229;
  --line2:#22282f;
  --txt0:#e6edf3;
  --txt1:#7d8590;
  --txt2:#6e7681;
  --green:#3fb950;
  --green-dim:rgba(63,185,80,0.12);
  --green-bright:#56d364;
  --clay:#f0883e;
  --clay-dim:rgba(240,136,62,0.12);
  --blue:#58a6ff;
  --blue-dim:rgba(88,166,255,0.12);
  --purple:#bc8cff;
  --purple-dim:rgba(188,140,255,0.12);
  --gold:#e3b341;
  --mono:'JetBrains Mono',ui-monospace,monospace;
  --display:'Inter Tight',system-ui,sans-serif;
  --sans:'Inter Tight',system-ui,sans-serif;
  --radius:0px;
  --radius-lg:0px;
}}"""

OLD_DOT = """.logo-dot{{
  width:8px;height:8px;border-radius:50%;
  background:var(--green);
  box-shadow:0 0 8px var(--green);
}}"""

NEW_DOT = """.logo-dot{{
  width:7px;height:7px;
  background:var(--green);
}}"""

OLD_HDR = """.hdr{{
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 0;
  border-bottom:1px solid var(--line2);
  position:sticky;top:0;
  background:rgba(8,12,16,0.92);
  backdrop-filter:blur(12px);
  z-index:100;
}}"""

NEW_HDR = """.hdr{{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 0;
  border-bottom:1px solid var(--line2);
  position:sticky;top:0;
  background:var(--bg0);
  z-index:100;
}}"""

EDITS = [
    (OLD_FONTS, NEW_FONTS, "font link -> Inter Tight + JetBrains Mono"),
    (OLD_ROOT,  NEW_ROOT,  ":root tokens -> console palette, square corners"),
    (OLD_DOT,   NEW_DOT,   ".logo-dot -> square indicator, glow removed"),
    (OLD_HDR,   NEW_HDR,   ".hdr -> solid bar, blur removed"),
]


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

    if MARKER in raw:
        print("SKIP: console theme already applied."); return

    out = raw
    for old, new, label in EDITS:
        o = old.replace("\n", nl)
        n = new.replace("\n", nl)
        c = out.count(o)
        if c != 1:
            print(f"ABORT: expected exactly 1 match for '{label}', found {c}.")
            print("       The template differs from the expected shape.")
            sys.exit(1)
        out = out.replace(o, n, 1)
        print(f"APPLY  {label}")

    if out == raw:
        print("ABORT: edits produced no change."); sys.exit(1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)
    print("Syntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")
    print("\nNEXT:")
    print("  python courtiq_engine.py site --output docs/index.html")
    print("  open docs/index.html   # check the picks page and a full R128 round")


if __name__ == "__main__":
    main()
