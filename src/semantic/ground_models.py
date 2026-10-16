"""Alternative ground-height predictors over the same map ground vertices.

Added for the IJGI revision (ground-model baselines and ablation). The
pipeline default stays the nearest-vertex model in shp_matcher.query_ground_z;
these are only used when explicitly requested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

IDW_K, IDW_P, SEARCH_R = 8, 2.0, 20.0
PLANE_K = 12


def predict_idw(tree, z, q, k=IDW_K, p=IDW_P, r=SEARCH_R):
    """Inverse-distance weighting of the k nearest vertices within r metres."""
    d, i = tree.query(q, k=k, distance_upper_bound=r)
    valid = np.isfinite(d)
    i = np.where(valid, i, 0)
    w = np.where(valid, 1.0 / np.maximum(d, 1e-3) ** p, 0.0)
    ws = w.sum(axis=1)
    out = np.full(len(q), np.nan)
    ok = ws > 0
    out[ok] = (w[ok] * z[i[ok]]).sum(axis=1) / ws[ok]
    return out


def predict_tin(xy, z, q):
    """Linear interpolation on a Delaunay TIN of the vertices.

    Vertices sharing an XY (within 1 cm) are averaged first, so stacked
    multi-level vertices collapse to one height. NaN outside the hull.
    Returns (pred, n_merged_duplicates).
    """
    key = np.round(xy, 2)
    dfv = pd.DataFrame({"x": key[:, 0], "y": key[:, 1], "z": z})
    agg = dfv.groupby(["x", "y"], as_index=False)["z"].mean()
    interp = LinearNDInterpolator(agg[["x", "y"]].to_numpy(), agg["z"].to_numpy())
    return interp(q), int(len(xy) - len(agg))


def predict_plane(tree, xy, z, q, k=PLANE_K, r=SEARCH_R):
    """Local least-squares plane on the k nearest vertices within r metres.

    Returns (pred, grade); NaN where fewer than 3 vertices or collinear.
    """
    d, idx = tree.query(q, k=k, distance_upper_bound=r)
    out = np.full(len(q), np.nan)
    grade = np.full(len(q), np.nan)
    for row in range(len(q)):
        m = np.isfinite(d[row])
        if m.sum() < 3:
            continue
        j = idx[row][m]
        A = np.column_stack([xy[j, 0] - q[row, 0], xy[j, 1] - q[row, 1], np.ones(m.sum())])
        if np.linalg.matrix_rank(A) < 3:
            continue
        coef, *_ = np.linalg.lstsq(A, z[j], rcond=None)
        out[row] = coef[2]
        grade[row] = float(np.hypot(coef[0], coef[1]))
    return out, grade
