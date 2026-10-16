"""Apply the viewer's tunnel reclassification (web/src/core/tunnel.ts) to a run.

Kept 'unknown' cells inside the tunnel macro cells (25 m) become 'tunnel'.
The tunnel macro-cell set is taken from the reference delivered run (P4) so
every variant is scored under exactly the same delivered definition.

  python eval/revision/apply_tunnel_reclass.py --ref P4/stages.npz --run X/stages.npz --out Y/stages.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MACRO = 25.0


def macro_ids(st):
    xyz = st["origin"] + (st["cells"] + 0.5) * float(st["voxel_size"])
    m = np.floor(xyz[:, :2] / MACRO).astype(np.int64)
    return m[:, 0] * 1_000_003 + m[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = dict(np.load(a.ref)); st = dict(np.load(a.run))
    tun = np.unique(macro_ids(ref)[ref["labels_final"].astype(str) == "tunnel"])
    lab = st["labels_final"].astype(str).astype(object)
    m = np.isin(macro_ids(st), tun) & (lab == "unknown") & st["keep"]
    lab[m] = "tunnel"
    st["labels_final"] = lab.astype(str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **st)
    print(f"{a.run}: {int(m.sum())} cells -> tunnel")


if __name__ == "__main__":
    main()
