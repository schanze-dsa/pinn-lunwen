#!/usr/bin/env python
"""Export PINN/FEM mirror surface samples to a Zemax Grid Sag DAT file.

The expected PINN visualization export is the text file written by
``viz/mirror_viz.py``:

    node_id x y z u_x u_y u_z |u| u_plane v_plane

The output follows the common OpticStudio Grid Sag format:

    nx ny dx dy unitflag xdec ydec
    sag dzdx dzdy d2zdxdy nodata
    ...

Coordinates and sag are assumed to be in mm unless scale factors are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_COLUMNS = [
    "node_id",
    "x",
    "y",
    "z",
    "u_x",
    "u_y",
    "u_z",
    "u_mag",
    "u_plane",
    "v_plane",
]

COLUMN_ALIASES = {
    "|u|": "u_mag",
    "mag": "u_mag",
    "magnitude": "u_mag",
    "ux": "u_x",
    "uy": "u_y",
    "uz": "u_z",
    "up": "u_plane",
    "vp": "v_plane",
}


def _normalize_name(name: str) -> str:
    key = name.strip().replace(",", "").replace("(", "").replace(")", "")
    key = key.replace("|u|", "u_mag")
    key = key.replace("-", "_")
    key = key.lower()
    return COLUMN_ALIASES.get(key, key)


def _split_data_line(line: str) -> List[str]:
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    return line.split()


def read_surface_table(path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    columns: Optional[List[str]] = None
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                marker = "columns:"
                low = line.lower()
                if marker in low:
                    raw_cols = line[low.index(marker) + len(marker) :].strip()
                    columns = [_normalize_name(c) for c in _split_data_line(raw_cols)]
                continue
            rows.append([float(v) for v in _split_data_line(line)])

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent column count in {path}")

    if columns is None:
        if width == len(DEFAULT_COLUMNS):
            columns = DEFAULT_COLUMNS[:]
        else:
            columns = [f"col{i}" for i in range(width)]

    if len(columns) != width:
        if width == len(DEFAULT_COLUMNS):
            columns = DEFAULT_COLUMNS[:]
        else:
            raise ValueError(
                f"Header has {len(columns)} columns but data has {width} columns in {path}"
            )

    col_map = {_normalize_name(name): idx for idx, name in enumerate(columns)}
    return np.asarray(rows, dtype=float), col_map


def require_column(col_map: Dict[str, int], name: str) -> int:
    key = _normalize_name(name)
    if key not in col_map:
        available = ", ".join(sorted(col_map))
        raise KeyError(f"Column '{name}' not found. Available columns: {available}")
    return col_map[key]


def _try_ckdtree(points: np.ndarray, queries: np.ndarray, k: int):
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return None
    tree = cKDTree(points)
    try:
        return tree.query(queries, k=min(k, len(points)), workers=-1)
    except TypeError:
        return tree.query(queries, k=min(k, len(points)))


def idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    power: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("No valid interpolation points")
    if len(points) == 1:
        return np.full(len(queries), float(values[0]), dtype=float)

    k_eff = max(1, min(int(k), len(points)))
    queried = _try_ckdtree(points, queries, k_eff)
    if queried is not None:
        dist, idx = queried
        if k_eff == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        return _weights_to_values(dist, values[idx], power)

    out = np.empty(len(queries), dtype=float)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        diff = queries[start:stop, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
        dist = np.sqrt(np.take_along_axis(dist2, idx, axis=1))
        out[start:stop] = _weights_to_values(dist, values[idx], power)
    return out


def _weights_to_values(dist: np.ndarray, vals: np.ndarray, power: float) -> np.ndarray:
    exact = dist <= 1.0e-12
    out = np.empty(dist.shape[0], dtype=float)
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        first_exact = np.argmax(exact[exact_rows], axis=1)
        out[exact_rows] = vals[exact_rows, first_exact]
    if np.any(~exact_rows):
        d = dist[~exact_rows]
        w = 1.0 / np.maximum(d, 1.0e-12) ** float(power)
        out[~exact_rows] = np.sum(w * vals[~exact_rows], axis=1) / np.sum(w, axis=1)
    return out


def finite_difference_derivatives(
    sag: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filled = sag.copy()
    filled[~valid] = np.nan
    dzdy, dzdx = np.gradient(filled, dy, dx)
    d2 = np.gradient(dzdx, dy, axis=0)
    for arr in (dzdx, dzdy, d2):
        arr[~np.isfinite(arr)] = 0.0
        arr[~valid] = 0.0
    return dzdx, dzdy, d2


def export_grid_sag(
    table: np.ndarray,
    col_map: Dict[str, int],
    *,
    output: Path,
    x_column: str,
    y_column: str,
    sag_column: str,
    nx: int,
    ny: int,
    coord_scale: float,
    sag_scale: float,
    inner_radius: Optional[float],
    outer_radius: Optional[float],
    padding: float,
    idw_neighbors: int,
    idw_power: float,
    write_derivatives: bool,
    summary_json: Optional[Path],
) -> Dict[str, object]:
    x = table[:, require_column(col_map, x_column)] * float(coord_scale)
    y = table[:, require_column(col_map, y_column)] * float(coord_scale)
    sag = table[:, require_column(col_map, sag_column)] * float(sag_scale)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(sag)
    x, y, sag = x[finite], y[finite], sag[finite]
    if len(x) < 3:
        raise ValueError("Need at least three valid surface samples")

    if outer_radius is not None:
        x_min, x_max = -float(outer_radius), float(outer_radius)
        y_min, y_max = -float(outer_radius), float(outer_radius)
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_pad = (x_max - x_min) * float(padding)
        y_pad = (y_max - y_min) * float(padding)
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    xdec = 0.5 * (x_min + x_max)
    ydec = 0.5 * (y_min + y_max)

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    queries = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    interp = idw_interpolate(
        np.column_stack([x, y]),
        sag,
        queries,
        k=idw_neighbors,
        power=idw_power,
    ).reshape(ny, nx)

    valid = np.isfinite(interp)
    radius = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    if inner_radius is not None:
        valid &= radius >= float(inner_radius)
    if outer_radius is not None:
        valid &= radius <= float(outer_radius)

    interp[~valid] = 0.0
    if write_derivatives:
        dzdx, dzdy, d2 = finite_difference_derivatives(interp, valid, dx, dy)
    else:
        dzdx = np.zeros_like(interp)
        dzdy = np.zeros_like(interp)
        d2 = np.zeros_like(interp)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as fp:
        fp.write(f"{nx:d} {ny:d} {dx:.12g} {dy:.12g} 0 {xdec:.12g} {ydec:.12g}\n")
        for row in range(ny):
            for col in range(nx):
                nodata = 0 if bool(valid[row, col]) else 1
                fp.write(
                    f"{interp[row, col]:.12e} "
                    f"{dzdx[row, col]:.12e} "
                    f"{dzdy[row, col]:.12e} "
                    f"{d2[row, col]:.12e} "
                    f"{nodata:d}\n"
                )

    summary = {
        "output": str(output),
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "xdec": xdec,
        "ydec": ydec,
        "valid_points": int(np.count_nonzero(valid)),
        "nodata_points": int(valid.size - np.count_nonzero(valid)),
        "sag_min": float(np.min(interp[valid])) if np.any(valid) else None,
        "sag_max": float(np.max(interp[valid])) if np.any(valid) else None,
        "sag_rms": float(math.sqrt(np.mean(interp[valid] ** 2))) if np.any(valid) else None,
        "x_column": _normalize_name(x_column),
        "y_column": _normalize_name(y_column),
        "sag_column": _normalize_name(sag_column),
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Surface sample txt/csv file.")
    parser.add_argument("--output", required=True, type=Path, help="Output Zemax Grid Sag .dat file.")
    parser.add_argument("--x-column", default="u_plane", help="Column used as local grid x coordinate.")
    parser.add_argument("--y-column", default="v_plane", help="Column used as local grid y coordinate.")
    parser.add_argument("--sag-column", default="u_z", help="Column used as sag deformation.")
    parser.add_argument("--nx", type=int, default=129, help="Grid points in x direction.")
    parser.add_argument("--ny", type=int, default=129, help="Grid points in y direction.")
    parser.add_argument("--coord-scale", type=float, default=1.0, help="Scale input coordinates to Zemax lens units.")
    parser.add_argument("--sag-scale", type=float, default=1.0, help="Scale input sag to Zemax lens units.")
    parser.add_argument("--inner-radius", type=float, default=None, help="Optional annular inner radius.")
    parser.add_argument("--outer-radius", type=float, default=None, help="Optional outer radius; also sets square grid bounds.")
    parser.add_argument("--padding", type=float, default=0.02, help="Bounding-box padding when outer radius is not set.")
    parser.add_argument("--idw-neighbors", type=int, default=8, help="Nearest samples used by IDW interpolation.")
    parser.add_argument("--idw-power", type=float, default=2.0, help="IDW distance power.")
    parser.add_argument("--no-derivatives", action="store_true", help="Write zero derivatives instead of finite differences.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    table, col_map = read_surface_table(args.input)
    summary = export_grid_sag(
        table,
        col_map,
        output=args.output,
        x_column=args.x_column,
        y_column=args.y_column,
        sag_column=args.sag_column,
        nx=args.nx,
        ny=args.ny,
        coord_scale=args.coord_scale,
        sag_scale=args.sag_scale,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        padding=args.padding,
        idw_neighbors=args.idw_neighbors,
        idw_power=args.idw_power,
        write_derivatives=not args.no_derivatives,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
