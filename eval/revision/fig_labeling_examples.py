"""Supplementary figure: labeling examples from the actual tools (English figure mode screenshots).

  python eval/revision/fig_labeling_examples.py --caps <dir> --out <png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"]})
PANELS = [
    ("s4_mixed", "(a) Mixed cell: guardrail with vegetation", "primary guardrail (green), second label vegetation (dashed)"),
    ("s4_post", "(b) Post re-check: cell on a pole carrying a sign", "answer 'part of a post: yes'"),
    ("s4_tunnelno", "(c) Tunnel re-check, other direction: low tunnel cell", "answer 'tunnel wall / base: no' (not a barrier)"),
    ("s4_amb90", "(d) Ambiguous object, judged again at the end", "post object mainly vegetation; second label post, purity 'about half'"),
    ("s4_obj32", "(e) Vehicle merged below a traffic light", "signal head 5–7 m; detached vehicle cells below (Section 4.5)"),
    ("s4_obj20", "(f) Mapped guardrail that is a low curb on site", "labeled guardrail; noted as a pedestrian-side curb (Table S8, B)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip", action="store_true", help="add the process strip of fig_labeling_tools on top")
    a = ap.parse_args()
    d = Path(a.caps)
    W = 170 / 25.4
    pw = W / 2
    ph = pw * 960 / 1600
    cap = 0.36
    hs = 0.62 if a.strip else 0.0
    H = 3 * (ph + cap) + hs + (0.08 if a.strip else 0)
    fig = plt.figure(figsize=(W, H))
    if a.strip:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from fig_labeling_tools import strip
        strip(fig.add_axes([0, 1 - hs / H, 1, hs / H]), tags=("", "", ""), panels=("\nexample: panel (a)", "\nexamples: (d)–(f)", "\nexamples: (b), (c)"))
    for i, (f, t, s) in enumerate(PANELS):
        r, c = divmod(i, 2)
        im = np.asarray(Image.open(d / f"{f}.png").convert("RGB"))[0:960]
        y = 1 - (hs + (0.08 if a.strip else 0) + (r + 1) * (ph + cap)) / H
        ax = fig.add_axes([c * 0.5 + 0.004, y, 0.492, ph / H]); ax.imshow(im); ax.axis("off")
        fig.text(c * 0.5 + 0.006, y + (ph + cap * 0.62) / H, t, fontsize=7.2, fontweight="bold", va="center")
        fig.text(c * 0.5 + 0.006, y + (ph + cap * 0.22) / H, s, fontsize=6.4, color="#52514e", va="center")
    fig.savefig(a.out, dpi=300)
    print(a.out)


if __name__ == "__main__":
    main()
