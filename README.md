# Training-free semantic labeling of MMS LiDAR with vector HD maps

Code and evaluation material for the article "Training-Free Semantic Labeling of Mobile LiDAR Using Vector HD Maps for
Driving Simulation" (ISPRS International Journal of Geo-Information, under revision).

The pipeline transfers the categories of a vector HD map (Korean precision road map, 11 SHP layers with 3D vertices)
onto unclassified mobile-mapping LiDAR and writes a labeled 0.5 m voxel scene. It trains no model.

## Contents

| Path | What it is |
|---|---|
| `src/` | Voxelization, map-derived ground model, elevation-gated labeling, vehicle filter, retention filters, object grouping |
| `run/run_build_scene_lidar.py` | Builds one scene; `--dump-stages` writes every intermediate stage used by the evaluation |
| `config/` | Grid, category dimensions (radii and elevation windows, Table 3), map layer specification, colors |
| `config/regions/deoksong.template.json` | Region file of the evaluated site with data paths removed |
| `eval/revision/` | Evaluation: ground-model baselines, reference sampling, Horvitz–Thompson scoring, occupancy and object scoring, ablation configs, post-hoc variants, figures, and the labeling viewer (`labeler/`) |
| `evaluation/` | Reference sample definition, blind labels, correction files, re-check lists, object-review labels and the final result tables |

## Data that are not included

- **Full MMS LiDAR** (Leica Pegasus:Two Ultimate, LAS files): acquired by GEOSTORY Co., Ltd. The reduced point clouds the
  pipeline consumed (each ROI, every 4th point) and the labeled 0.5 m voxel scenes of both sites are published with the
  permission of GEOSTORY in the Zenodo data record of this article (see its `README_data.md`).
- **Precision road map**: the article used NGII sections SEC001 (Deoksong) and SEC203 (Road-86), version ver_1.2
  (created 29 December 2023), HDMap_UTM52N (ellipsoidal heights). NGII distributes it (map.ngii.go.kr) to
  registered users under its terms of use, which prohibit redistribution and transfer abroad, so map
  geometries are not included here.
- Survey photographs and the point crops shown in the labeling viewer are derived from the LiDAR and photographs and are
  therefore not included either.

## Installation

Python 3.10 or later; the results were produced with Python 3.13 and the package versions pinned in
`requirements.txt`.

```
pip install -r requirements.txt
```

## Running

1. Copy `config/regions/deoksong.template.json` to `config/regions/<name>.json` and set the paths to your LAS files and
   SHP directory. With the reduced LiDAR of the Zenodo record, point `lidar_globs` to its
   `VoxelLabel_<site>_LiDAR_ROI_stride4_part*.laz` files and set `"lidar_stride": 1`: those files already contain every
   4th point, so a stride of 4 would thin them again.
2. Build the scene and dump the stages. Run from the repository root with the root on the Python path (`PYTHONPATH=.` on
   Linux/macOS, `set PYTHONPATH=.` on Windows). The evaluated configuration of the article is:

```
PYTHONPATH=. python run/run_build_scene_lidar.py --region <name> --measure-only --dimensions eval/revision/configs/dims_submitted_sign2m.json --no-connected-pole-protection --dump-stages out/stages.npz
```

   With the Zenodo LiDAR and the map, this reproduces the cells, labels, filters and objects of the evaluated run exactly.

### Pipeline options

- Connected pole protection is **on by default**: a vehicle candidate protected by a tall structure above it keeps its
  protection only if it is 26-connected, through pole-category cells, to that structure; otherwise it loses the pole
  label and is kept as unknown occupancy (Supplementary Text S3).
- `--no-connected-pole-protection` reproduces the configuration evaluated in the article exactly (column test only).
- `--dimensions eval/revision/configs/dims_submitted_sign2m.json` reproduces the evaluated matching parameters
  (sign radius 2.0 m).

3. Score against the reference sample (the cell rows refer to the occupied cells of this run, so the same inputs and
   parameters are required). Write the sample directory from `evaluation/reference_sample.json` first. Without the map,
   use `VoxelLabel_Deoksong-SEC001_stages_delivered.npz` from the Zenodo record as `out/stages_delivered.npz`; this
   reproduces the cell-level and occupancy tables of the article exactly (`eval_objects_delivered.py` also needs the map
   layers, `--hdmap`).

```
python eval/revision/make_sample_dir.py --reference evaluation/reference_sample.json --out out/sample
python eval/revision/eval_sample.py --sample-dir out/sample --labels evaluation/labels_blind_FINAL.json --corrections evaluation/corrections_1_post_recheck.json evaluation/corrections_2_tunnel_guardrail_predicted.json evaluation/corrections_3_tunnel_guardrail_symmetric.json --post-labels evaluation/post_recheck_answers.json --definition delivered --runs final=out/stages_delivered.npz --out-dir out/eval
```

`eval/revision/apply_tunnel_reclass.py` derives the tunnel category of the delivered scene, and
`eval/revision/build_final_tables.py` assembles the tables.

4. Ground-model comparison (Section 4.1, Table S5; needs the LiDAR and the map): set `VOXELLABEL_DEOKSONG_DIR` or
   `VOXELLABEL_ROAD86_DIR` to the site folder holding `HDMap_UTM52N_타원체고/` and the `SURV*/TRACK*/Laser01/*.las` files
   (the NGII delivery layout), then run
   `python eval/revision/ground_baselines.py --region deoksong --source raw --out-dir out/ground` (or `--region road86`).

## Evaluation material

- `reference_sample.json`: every drawn cell with its stratum, stratum size `N_h`, row index and cell centre (UTM 52N).
- `labels_blind_FINAL.json`: blind labels, frozen before any re-check. They are never edited.
- `corrections_*.json`: later yes/no re-checks, applied at scoring time in the numbered order. Results are reported
  before and after them.
- `object_review_*`: blind review of all 91 output objects and the grouped reviewer notes.
- `road86_ground_z_errors.csv`: per-cell ground-model error at Road-86 (nearest-vertex model; median absolute error
  2.8 cm, Section 4.1 and Table S5). The modeled and reference heights are omitted because the modeled height is
  interpolated from the map vertices; the error, cell position, point count and height spread are kept.
- `road86_spotcheck_cell_edits.json`: the 256 operator-labeled cells of the Road-86 spot check (Table S2), in the
  viewer's cell-edit format; scoring box (339974, 4164305)–(340054, 4164385), UTM 52N.

## License

MIT (see `LICENSE`). Cite the article when you use the code or the evaluation material.
