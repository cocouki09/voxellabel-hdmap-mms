"""Turn evaluation/reference_sample.json into the sample directory the scoring scripts read.

eval_sample.py and eval_occupancy.py read <sample dir>/KEY_do_not_open_while_labeling.json
(the key written by sample_cells.py when the sample was drawn). The release ships the same
key as evaluation/reference_sample.json; this writes it back under the expected name.

  python eval/revision/make_sample_dir.py --reference evaluation/reference_sample.json --out out/sample
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="evaluation/reference_sample.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.reference, encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {"seed": ref["seed"], "voxel_size": ref["voxel_size"], "key": ref["samples"]}
    (out / "KEY_do_not_open_while_labeling.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(ref['samples'])} samples -> {out / 'KEY_do_not_open_while_labeling.json'}")


if __name__ == "__main__":
    main()
