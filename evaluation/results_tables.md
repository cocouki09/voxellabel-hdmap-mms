# Final tables (W04_FINAL_20261015)

Sample: 1090 drawn, 695 used, 362 not labeled (beyond quota), 33 excluded (cannot judge / isolated returns).

## Table 6 — cell level, delivered definitions (final labels incl. corrections)

| Category | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Guardrail | 0.88 [0.80, 0.95] | 0.62 [0.47, 0.81] | 0.73 [0.61, 0.85] | 0.57 [0.44, 0.74] |
| Traffic light | 0.25 [0.15, 0.37] | 1.00 [1.00, 1.00] | 0.41 [0.26, 0.54] | 0.25 [0.15, 0.37] |
| Sign | 0.12 [0.05, 0.20] | 0.17 [0.05, 1.00] | 0.14 [0.05, 0.29] | 0.08 [0.02, 0.17] |
| Post / street light | 0.41 [0.29, 0.54] | 0.30 [0.11, 1.00] | 0.35 [0.17, 0.67] | 0.21 [0.09, 0.50] |
| Overpass (height barrier) | 1.00 [1.00, 1.00] | 0.05 [0.02, 0.15] | 0.09 [0.04, 0.26] | 0.05 [0.02, 0.15] |
| Tunnel | 0.90 [0.84, 0.95] | 0.97 [0.93, 0.99] | 0.94 [0.90, 0.97] | 0.88 [0.81, 0.94] |
| Macro average | | | 0.44 [0.39, 0.52] | 0.34 [0.30, 0.41] |

## Table 6 (reference) — cell level, strict definitions

| Category | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Guardrail | 0.85 [0.76, 0.93] | 0.59 [0.45, 0.78] | 0.70 [0.58, 0.82] | 0.54 [0.41, 0.70] |
| Traffic light | 0.25 [0.15, 0.37] | 0.60 [0.42, 0.77] | 0.36 [0.22, 0.48] | 0.22 [0.12, 0.31] |
| Sign | 0.10 [0.03, 0.19] | 0.13 [0.03, 0.67] | 0.12 [0.03, 0.25] | 0.06 [0.02, 0.14] |
| Post / street light | 0.19 [0.08, 0.29] | 0.16 [0.05, 1.00] | 0.17 [0.07, 0.42] | 0.10 [0.04, 0.27] |
| Overpass (height barrier) | 0.00 [0.00, 0.00] | – | – | 0.00 [0.00, 0.00] |
| Macro average | | | 0.34 [0.28, 0.42] | 0.18 [0.15, 0.24] |

## Effect of the tunnel–guardrail re-check (delivered)

| Category | P / R / F1 before | after 40 predicted-guardrail cells | after both directions (40 + 27) |
|---|---|---|---|
| Guardrail | 0.20 / 0.36 / 0.26 | 0.88 / 0.71 / 0.79 | 0.88 / 0.62 / 0.73 |
| Traffic light | 0.25 / 1.00 / 0.41 | 0.25 / 1.00 / 0.41 | 0.25 / 1.00 / 0.41 |
| Sign | 0.12 / 0.17 / 0.14 | 0.12 / 0.17 / 0.14 | 0.12 / 0.17 / 0.14 |
| Post / street light | 0.41 / 0.30 / 0.35 | 0.41 / 0.30 / 0.35 | 0.41 / 0.30 / 0.35 |
| Overpass (height barrier) | 1.00 / 0.05 / 0.09 | 1.00 / 0.05 / 0.09 | 1.00 / 0.05 / 0.09 |
| Tunnel | 0.93 / 0.87 / 0.90 | 0.93 / 0.96 / 0.94 | 0.90 / 0.97 / 0.94 |
| Macro-F1 | 0.36 [0.31, 0.44] | 0.45 [0.41, 0.53] | 0.44 [0.39, 0.52] |

## Sensitivities (delivered)

| Scoring | Macro-F1 | Macro-IoU |
|---|---|---|
| Primary label | 0.44 [0.39, 0.52] | 0.34 [0.30, 0.41] |
| Lenient (either label) | 0.44 | |
| Ambiguous cells excluded | 0.45 [0.40, 0.53] | 0.35 [0.31, 0.42] |
| Second label when primary is noise | 0.44 [0.40, 0.52] | 0.34 [0.30, 0.41] |

## Table 7 — ablation and sensitivity (delivered; P / R)

| Variant | Guardrail | Traffic light | Sign | Post / street light | Tunnel | Macro-F1 |
|---|---|---|---|---|---|---|
| Evaluated configuration (sign 2.0 m) | 0.88 / 0.62 | 0.25 / 1.00 | 0.12 / 0.17 | 0.41 / 0.30 | 0.90 / 0.97 | 0.44 [0.39, 0.52] |
| Constant ground model | 0.00 / 0.00 | – / 0.00 | – / 0.00 | – / 0.00 | 0.99 / 0.40 | 0.09 [0.08, 0.11] |
| IDW ground model | 0.90 / 0.62 | 0.25 / 1.00 | 0.12 / 0.17 | 0.41 / 0.30 | 0.90 / 0.97 | 0.44 [0.40, 0.52] |
| Elevation gate off | 0.35 / 0.63 | 0.67 / 0.21 | 0.15 / 0.25 | 0.20 / 0.17 | 0.00 / 0.00 | 0.22 [0.16, 0.30] |
| One radius (2.0 m) for all categories | 0.66 / 0.87 | 0.52 / 0.87 | 0.13 / 0.17 | 0.60 / 0.30 | 0.95 / 0.95 | 0.49 [0.44, 0.59] |
| Vehicle filter off | 0.88 / 0.62 | 0.20 / 1.00 | 0.12 / 0.17 | 0.41 / 0.30 | 0.90 / 0.97 | 0.43 [0.38, 0.51] |
| Pole protection off | 0.88 / 0.62 | 0.25 / 0.62 | 0.10 / 0.14 | 0.31 / 0.22 | 0.90 / 0.97 | 0.41 [0.37, 0.49] |
| Sign radius 1.0 m | 0.88 / 0.62 | 0.25 / 1.00 | 0.24 / 0.17 | 0.34 / 0.31 | 0.90 / 0.97 | 0.45 [0.40, 0.53] |
| All radii ×0.75 | 0.90 / 0.52 | 0.39 / 1.00 | 0.16 / 0.17 | 0.57 / 0.29 | 0.89 / 0.98 | 0.46 [0.41, 0.56] |
| All radii ×1.25 | 0.81 / 0.66 | 0.22 / 1.00 | 0.08 / 0.17 | 0.30 / 0.31 | 0.91 / 0.96 | 0.42 [0.38, 0.50] |
| Windows narrowed 0.25 m | 0.98 / 0.53 | 0.25 / 1.00 | 0.13 / 0.17 | 0.40 / 0.30 | 0.90 / 0.97 | 0.43 [0.39, 0.51] |
| Windows widened 0.25 m | 0.66 / 0.72 | 0.25 / 1.00 | 0.12 / 0.17 | 0.39 / 0.30 | 0.90 / 0.97 | 0.43 [0.39, 0.51] |
| Post hoc: overhead-sign window | 0.88 / 0.62 | 0.20 / 0.71 | 0.21 / 0.37 | 0.40 / 0.29 | 0.90 / 0.97 | 0.44 [0.39, 0.53] |
| Post hoc: exclude tunnel signals (C1 type 9, 4 objects) | 0.88 / 0.62 | 0.75 / 1.00 | 0.12 / 0.17 | 0.41 / 0.30 | 0.90 / 0.98 | 0.52 [0.47, 0.60] |
| Post hoc: exclude map types C3-8, C1-9, C5-1 (10 objects) | 0.73 / 0.13 | 0.75 / 1.00 | 0.12 / 0.17 | 0.41 / 0.30 | 0.82 / 0.98 | 0.41 [0.36, 0.48] |

## Vehicle filter and unknown composition (delivered, final labels)

- removed ≈493 cells; removal precision 0.73 [0.54, 0.88]; structure among removed 0.27 [0.12, 0.46]
- residual vehicle cells ≈4454 [392, 10351]
- unknown cells (HT estimate) ≈98664; composition: vegetation 71.3%, viaduct 9.8%, guardrail 4.1%, vehicle 4.1%, other_structure 3.8%, building_or_wall 3.1%, street_light 1.9%, ground_or_road_surface 1.9%

Intra-rater (blind first-pass labels, before any correction): 49 pairs, agreement 0.878, κ 0.855 [0.738, 0.951]

## Static-occupancy accuracy (share of output cells that are static solid content)

stages ../output/ijgi_revision/W07_RUNS/_delivered/A0/stages.npz; non-static = ['noise_or_empty', 'vehicle']; cannot_judge excluded

| Output class | Sample cells | Static-occupancy accuracy [95% CI] |
|---|---:|---|
| guardrail | 59 | 1.000 [1.000, 1.000] |
| height_barrier | 39 | 0.974 [0.923, 1.000] |
| sign | 59 | 1.000 [1.000, 1.000] |
| street_light | 60 | 0.983 [0.950, 1.000] |
| traffic_light | 60 | 0.983 [0.950, 1.000] |
| tunnel | 217 | 0.997 [0.992, 1.000] |
| unknown | 78 | 0.948 [0.888, 0.990] |
| all kept cells | 572 | 0.980 [0.958, 0.995] |


## Object-level review — delivered definitions

Objects 91; with a category 87.

## All output objects

| Category | Mapped in ROI | Output objects | Reviewed | Correct (precision, Wilson 95% CI) | Correct incl. shared pole | Purity of correct: mostly / half / mostly other / not given | Object recall (correct / mapped) |
|---|---:|---:|---:|---|---|---|---|
| guardrail | 33 | 33 | 33 | 33/33 = 1.00 [0.90, 1.00] | 33/33 = 1.00 [0.90, 1.00] | 30 / 0 / 0 / 3 | 33/33 = 1.00 |
| traffic_light | 26 | 26 | 26 | 15/26 = 0.58 [0.39, 0.74] | 21/26 = 0.81 [0.62, 0.91] | 10 / 4 / 0 / 1 | 15/26 = 0.58 |
| sign | 25 | 14 | 13 | 2/13 = 0.15 [0.04, 0.42] | 6/13 = 0.46 [0.23, 0.71] | 1 / 1 / 0 / 0 | 2/25 = 0.08 |
| street_light | 16 | 16 | 13 | 13/13 = 1.00 [0.77, 1.00] | 13/13 = 1.00 [0.77, 1.00] | 8 / 3 / 0 / 2 | 13/16 = 0.81 |
| height_barrier | 2 | 2 | 2 | 2/2 = 1.00 [0.34, 1.00] | 2/2 = 1.00 [0.34, 1.00] | 2 / 0 / 0 / 0 | 2/2 = 1.00 |
| **all** | 102 | 91 | 87 | 65/87 = 0.75 [0.65, 0.83] | 75/87 = 0.86 | | 65/102 = 0.64 |

Wrong objects were reviewed as: traffic_light: {'tunnel_structure': 4, 'street_light': 5, 'noise_or_empty': 1, 'sign': 1}; sign: {'street_light': 4, 'noise_or_empty': 3, 'vegetation': 2, 'vehicle': 1, 'cannot_judge': 1}

## Map Type filter (C3-8, C1-9, C5-1 removed)

| Category | Mapped in ROI | Output objects | Reviewed | Correct (precision, Wilson 95% CI) | Correct incl. shared pole | Purity of correct: mostly / half / mostly other / not given | Object recall (correct / mapped) |
|---|---:|---:|---:|---|---|---|---|
| guardrail | 29 | 29 | 29 | 29/29 = 1.00 [0.88, 1.00] | 29/29 = 1.00 [0.88, 1.00] | 26 / 0 / 0 / 3 | 29/29 = 1.00 |
| traffic_light | 22 | 22 | 22 | 15/22 = 0.68 [0.47, 0.84] | 21/22 = 0.95 [0.78, 0.99] | 10 / 4 / 0 / 1 | 15/22 = 0.68 |
| sign | 25 | 14 | 13 | 2/13 = 0.15 [0.04, 0.42] | 6/13 = 0.46 [0.23, 0.71] | 1 / 1 / 0 / 0 | 2/25 = 0.08 |
| street_light | 16 | 16 | 13 | 13/13 = 1.00 [0.77, 1.00] | 13/13 = 1.00 [0.77, 1.00] | 8 / 3 / 0 / 2 | 13/16 = 0.81 |
| height_barrier | 0 | 0 | 0 | 0/0 = nan [nan, nan] | 0/0 = nan [nan, nan] | 0 / 0 / 0 / 0 | 0/0 = nan |
| **all** | 92 | 81 | 77 | 59/77 = 0.77 [0.66, 0.85] | 69/77 = 0.90 | | 59/92 = 0.64 |

Wrong objects were reviewed as: traffic_light: {'street_light': 5, 'noise_or_empty': 1, 'sign': 1}; sign: {'street_light': 4, 'noise_or_empty': 3, 'vegetation': 2, 'vehicle': 1, 'cannot_judge': 1}

Objects without a category answer: [19, 50, 71, 90]

