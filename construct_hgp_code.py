#!/usr/bin/env python3
"""Construct a hypergraph product (HGP) quantum code with qldpc.

An HGP code is a CSS code built from two classical linear codes A and B with
parity-check matrices H1 and H2:

    Hx = [ H1 ⊗ I(n2)  |  I(m1) ⊗ H2.T ]
    Hz = [-I(n1) ⊗ H2  |  H1.T ⊗ I(m2) ]

where (m_i, n_i) = H_i.shape.  If B is omitted, the construction uses A twice
(the homological square).

Examples:
    python construct_hgp_code.py
    python construct_hgp_code.py --code-a hamming:3
    python construct_hgp_code.py --code-a repetition:5 --code-b repetition:4
    python construct_hgp_code.py --code-a 'cyclic:7,1+x+x**3' --code-b hamming:3
    python construct_hgp_code.py --code-a mackay:16
    python construct_hgp_code.py --code-a mackay:20
    python construct_hgp_code.py --code-a mackay:24
    python construct_hgp_code.py --code-a semitopo:0
    python construct_hgp_code.py --code-a random:10,6 --seed 0 --no-distance
    python construct_hgp_code.py --code-a hamming:3 --save-dir out/hgp_hamming
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy
from qldpc.codes import (
    ClassicalCode,
    CyclicCode,
    ExtendedHammingCode,
    HammingCode,
    HGPCode,
    ReedMullerCode,
    RepetitionCode,
    RingCode,
    SimplexCode,
)
from qldpc.objects import Pauli

# (3,4)-regular MacKay–Neal seeds for the constant-rate HGP family in Table I of
# Roffe et al., Phys. Rev. Research 2, 043423 (2020).  Each H is full rank with
# girth > 4, so H^T encodes nothing (d^T = ∞).  The length-16 matrix is the
# published seed from Example 3.2 of arXiv:2202.01702; 20 and 24 are
# ensemble-equivalent (3,4)-regular codes with the table's [n,k,d].
MACKAY_H = {
    16: np.array(
        [
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        ],
        dtype=int,
    ),
    20: np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    24: np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
}

# (2,3)-LDPC parent of Roffe et al. semitopological codes (Table II / Fig. 2 of
# Phys. Rev. Research 2, 043423): H = [[1,1,1],[1,1,1]], a [3,2,2] code.
# Edge-augmenting every Tanner edge with a length-g repetition chain, then
# taking the hypergraph product, yields HGP(C_H^{*g}).  g=0 is [[13,5,2]].
PARENT_23_H = np.array([[1, 1, 1], [1, 1, 1]], dtype=int)


@dataclass(frozen=True)
class ClassicalSummary:
    spec: str
    name: str
    num_bits: int
    num_checks: int
    dimension: int
    distance: int | float | None
    check_shape: tuple[int, int]
    row_weights: tuple[int, ...]
    col_weights: tuple[int, ...]


def parse_seed(spec: str, field: int, seed: int | None) -> ClassicalCode:
    """Build a classical seed code from a compact spec string.

    Specs:
        repetition:N
        ring:N
        hamming:M              Hamming code of rank M  (binary length 2^M - 1)
        extended-hamming:M
        simplex:K
        reed-muller:R,M
        cyclic:N,POLY          e.g. cyclic:7,1+x+x**3
        mackay:N               (3,4)-regular MacKay seed; N=16,20,24 → [[400,16,6]], [[625,25,8]], [[900,36,10]]
        semitopo:G             (2,3)-LDPC parent with edge-augmentation g (g=0,1,2 → [[13,5,2]], [[145,5,6]], [[421,5,10]])
        random:N,M             N bits, M checks
        matrix:PATH            integer parity-check matrix (whitespace / commas)
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid seed spec {spec!r}. Expected family:params, e.g. hamming:3"
        )
    family, payload = spec.split(":", 1)
    family = family.strip().lower()
    payload = payload.strip()

    if family in {"repetition", "rep"}:
        return RepetitionCode(_one_int(payload, family), field=field)
    if family == "ring":
        return RingCode(_one_int(payload, family), field=field)
    if family == "hamming":
        return HammingCode(_one_int(payload, family), field=field)
    if family in {"extended-hamming", "ehamming"}:
        if field != 2:
            raise ValueError("extended-hamming is only defined over GF(2)")
        return ExtendedHammingCode(_one_int(payload, family))
    if family == "simplex":
        return SimplexCode(_one_int(payload, family), field=field)
    if family in {"reed-muller", "rm"}:
        order, size = _two_ints(payload, family)
        return ReedMullerCode(order, size, field=field)
    if family == "cyclic":
        return _cyclic_code(payload, field)
    if family == "mackay":
        return _mackay_code(payload, field)
    if family in {"semitopo", "augmented"}:
        return _semitopo_seed(payload, field)
    if family == "random":
        bits, checks = _two_ints(payload, family)
        return ClassicalCode.random(bits, checks, field=field, seed=seed)
    if family == "matrix":
        return ClassicalCode(_load_matrix(Path(payload)), field=field)

    raise ValueError(
        f"Unknown classical family {family!r}. "
        "Use repetition, ring, hamming, extended-hamming, simplex, "
        "reed-muller, cyclic, mackay, semitopo, random, or matrix."
    )


def _one_int(payload: str, family: str) -> int:
    try:
        return int(payload)
    except ValueError as exc:
        raise ValueError(f"{family} expects an integer, got {payload!r}") from exc


def _two_ints(payload: str, family: str) -> tuple[int, int]:
    parts = [p.strip() for p in payload.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{family} expects two integers, got {payload!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{family} expects two integers, got {payload!r}") from exc


def _cyclic_code(payload: str, field: int) -> CyclicCode:
    if "," not in payload:
        raise ValueError(
            "cyclic expects N,POLY, e.g. cyclic:7,1+x+x**3 "
            "(use ** for powers; ^ is also accepted)"
        )
    bits_str, poly_str = payload.split(",", 1)
    bits = int(bits_str.strip())
    poly_str = poly_str.strip().replace("^", "**")
    poly = sympy.sympify(poly_str)
    if not isinstance(poly, sympy.Basic):
        raise ValueError(f"Could not parse polynomial {poly_str!r}")
    return CyclicCode(bits, poly, field=field)


def _mackay_code(payload: str, field: int) -> ClassicalCode:
    bits = _one_int(payload, "mackay")
    if bits not in MACKAY_H:
        raise ValueError(
            "Built-in MacKay seeds are mackay:16, mackay:20, and mackay:24 "
            f"(got mackay:{bits}). For other lengths, supply a matrix file."
        )
    if field != 2:
        raise ValueError("mackay seeds are binary; use --field 2")
    return ClassicalCode(MACKAY_H[bits], field=2)


def _semitopo_seed(payload: str, field: int) -> ClassicalCode:
    """(2,3)-LDPC parent, optionally edge-augmented by g (Roffe et al. 2020)."""
    g = _one_int(payload, "semitopo")
    if g < 0:
        raise ValueError("semitopo:G requires G >= 0")
    if field != 2:
        raise ValueError("semitopo seeds are binary; use --field 2")
    return ClassicalCode(_edge_augment(PARENT_23_H, g), field=2)


def _edge_augment(matrix: np.ndarray, g: int) -> np.ndarray:
    """Replace every Tanner-graph edge with a length-g repetition path.

    For g=0 the parent matrix is returned unchanged.  For g>=1 the edge
    between parent check i and parent bit j becomes the path

        v_j -- u_1 -- b_1 -- u_2 -- b_2 -- ... -- u_g -- b_g -- u_i

    so every new data node has degree 2 (as required by Appendix A of
    arXiv:2005.07016).  The g×g chain block is therefore identity plus
    subdiagonal, not superdiagonal: a superdiagonal chain leaves the first
    new bit at degree 1 and the last new check at weight 1, which for g>=2
    forces those bits to 0 and drops the parent parity checks.
    """
    parent = np.asarray(matrix, dtype=int)
    if g == 0:
        return parent.copy()

    n_checks, n_bits = parent.shape
    edges = [(i, j) for i in range(n_checks) for j in range(n_bits) if parent[i, j]]
    n_edges = len(edges)
    augmented = np.zeros((n_checks + g * n_edges, n_bits + g * n_edges), dtype=int)

    for edge_index, (check, bit) in enumerate(edges):
        chain_check0 = n_checks + edge_index * g
        chain_bit0 = n_bits + edge_index * g
        for step in range(g):
            augmented[chain_check0 + step, chain_bit0 + step] = 1
            if step > 0:
                augmented[chain_check0 + step, chain_bit0 + step - 1] = 1
        augmented[chain_check0, bit] = 1
        augmented[check, chain_bit0 + g - 1] = 1
    return augmented


def _load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Parity-check matrix file not found: {path}")
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty matrix file: {path}")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([int(tok) for tok in line.replace(",", " ").split()])
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Ragged matrix in {path}: row lengths {sorted(widths)}")
    return np.array(rows, dtype=int)


def summarize_classical(
    code: ClassicalCode, spec: str, *, compute_distance: bool
) -> ClassicalSummary:
    matrix = np.asarray(code.matrix, dtype=int)
    distance: int | float | None = None
    if compute_distance:
        distance = code.get_distance()
    return ClassicalSummary(
        spec=spec,
        name=code.name,
        num_bits=code.num_bits,
        num_checks=code.num_checks,
        dimension=code.dimension,
        distance=distance,
        check_shape=tuple(matrix.shape),
        row_weights=tuple(int(w) for w in np.count_nonzero(matrix, axis=1)),
        col_weights=tuple(int(w) for w in np.count_nonzero(matrix, axis=0)),
    )


def check_weights(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dense = np.asarray(matrix, dtype=int)
    return np.count_nonzero(dense, axis=1), np.count_nonzero(dense, axis=0)


def format_params(n: int, k: int, d: int | float | None) -> str:
    if d is None:
        return f"[[{n}, {k}, ?]]"
    if isinstance(d, float) and np.isnan(d):
        return f"[[{n}, {k}, nan]]"
    return f"[[{n}, {k}, {int(d)}]]"


def construct_hgp(
    code_a: ClassicalCode,
    code_b: ClassicalCode | None,
    field: int,
) -> HGPCode:
    if code_b is None:
        return HGPCode(code_a, field=field)
    return HGPCode(code_a, code_b, field=field)


def report(
    code: HGPCode,
    seed_a: ClassicalSummary,
    seed_b: ClassicalSummary,
    *,
    compute_distance: bool,
    bound: int | None,
) -> dict:
    hx = np.asarray(code.matrix_x, dtype=int)
    hz = np.asarray(code.matrix_z, dtype=int)
    hx_row, hx_col = check_weights(hx)
    hz_row, hz_col = check_weights(hz)

    n, k = code.num_qudits, code.dimension
    distance = distance_x = distance_z = None
    if compute_distance:
        if bound:
            distance = code.get_distance(bound=bound)
            distance_x = code.get_distance(Pauli.X, bound=bound)
            distance_z = code.get_distance(Pauli.Z, bound=bound)
        else:
            distance = code.get_distance()
            distance_x = code.get_distance(Pauli.X)
            distance_z = code.get_distance(Pauli.Z)

    css_ok = not np.any(code.matrix_x @ code.matrix_z.T)

    print("Hypergraph product code")
    print(f"  field:            GF({code.field.order})")
    print(f"  seeds:            A={seed_a.spec}  B={seed_b.spec}")
    print(f"  parameters:       {format_params(n, k, distance)}")
    if compute_distance:
        label = "upper bound" if bound else "exact"
        print(f"  distance ({label}): d={distance}, dX={distance_x}, dZ={distance_z}")
    print(f"  data qudits n:    {n}")
    print(f"  logical qudits k: {k}")
    print(f"  X checks / Z checks: {code.num_checks_x} / {code.num_checks_z}")
    print(f"  max check weight: {code.get_weight()}")
    print(f"  Hx shape:         {hx.shape}   row weights {int(hx_row.min())}–{int(hx_row.max())}")
    print(f"  Hz shape:         {hz.shape}   row weights {int(hz_row.min())}–{int(hz_row.max())}")
    print(f"  CSS commutativity Hx Hz^T = 0: {css_ok}")
    print("  HGP sectors (data / check counts):")
    print(f"    (0,0) data:  {int(code.sector_size[0, 0])}")
    print(f"    (0,1) Z-chk: {int(code.sector_size[0, 1])}")
    print(f"    (1,0) X-chk: {int(code.sector_size[1, 0])}")
    print(f"    (1,1) data:  {int(code.sector_size[1, 1])}")
    print()
    _print_seed("code A", seed_a)
    _print_seed("code B", seed_b)

    return {
        "field": int(code.field.order),
        "n": int(n),
        "k": int(k),
        "d": _json_number(distance),
        "d_x": _json_number(distance_x),
        "d_z": _json_number(distance_z),
        "distance_is_bound": bool(bound) if compute_distance else None,
        "num_checks_x": int(code.num_checks_x),
        "num_checks_z": int(code.num_checks_z),
        "max_check_weight": int(code.get_weight()),
        "hx_shape": list(hx.shape),
        "hz_shape": list(hz.shape),
        "hx_row_weight_min": int(hx_row.min()) if len(hx_row) else None,
        "hx_row_weight_max": int(hx_row.max()) if len(hx_row) else None,
        "hz_row_weight_min": int(hz_row.min()) if len(hz_row) else None,
        "hz_row_weight_max": int(hz_row.max()) if len(hz_row) else None,
        "css_commute": bool(css_ok),
        "sector_size": code.sector_size.astype(int).tolist(),
        "code_a": _seed_dict(seed_a),
        "code_b": _seed_dict(seed_b),
    }


def _print_seed(label: str, seed: ClassicalSummary) -> None:
    print(f"  {label}: {seed.spec}")
    print(
        f"    classical params: [{seed.num_bits}, {seed.dimension}"
        + (f", {seed.distance}]" if seed.distance is not None else ", ?]")
    )
    print(f"    H shape: {seed.check_shape}")
    if seed.row_weights:
        print(
            f"    row weights: {min(seed.row_weights)}–{max(seed.row_weights)}, "
            f"col weights: {min(seed.col_weights)}–{max(seed.col_weights)}"
        )


def _seed_dict(seed: ClassicalSummary) -> dict:
    return {
        "spec": seed.spec,
        "name": seed.name,
        "n": seed.num_bits,
        "k": seed.dimension,
        "d": _json_number(seed.distance),
        "num_checks": seed.num_checks,
        "check_shape": list(seed.check_shape),
        "row_weight_min": min(seed.row_weights) if seed.row_weights else None,
        "row_weight_max": max(seed.row_weights) if seed.row_weights else None,
    }


def _json_number(value: int | float | None) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    if np.isnan(number):
        return None
    return int(number) if number.is_integer() else number


def save_code(code: HGPCode, out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hx = np.asarray(code.matrix_x, dtype=int)
    hz = np.asarray(code.matrix_z, dtype=int)
    np.savetxt(out_dir / "hx.txt", hx, fmt="%d")
    np.savetxt(out_dir / "hz.txt", hz, fmt="%d")
    np.save(out_dir / "hx.npy", hx)
    np.save(out_dir / "hz.npy", hz)

    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=int)
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=int)
    np.savetxt(out_dir / "logical_x.txt", logical_x, fmt="%d")
    np.savetxt(out_dir / "logical_z.txt", logical_z, fmt="%d")

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote Hx, Hz, logical operators, and metadata to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a hypergraph product CSS code with qldpc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--code-a",
        default="hamming:3",
        help="Classical seed A. Default: hamming:3  ([7,4,3] Hamming).",
    )
    parser.add_argument(
        "--code-b",
        default=None,
        help="Classical seed B. Default: same as A.",
    )
    parser.add_argument(
        "--field",
        type=int,
        default=2,
        help="Prime-power field order (default: 2, qubits).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for random:N,M seeds.")
    parser.add_argument(
        "--no-distance",
        action="store_true",
        help="Skip distance calculation (useful for large or random seeds).",
    )
    parser.add_argument(
        "--bound",
        type=int,
        default=None,
        metavar="TRIALS",
        help="Upper-bound quantum distance with this many QDistRnd trials instead of exact d.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Write Hx, Hz, logical operators, and metadata.json here.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compute_distance = not args.no_distance

    try:
        code_a = parse_seed(args.code_a, args.field, args.seed)
        code_b = (
            parse_seed(args.code_b, args.field, None if args.seed is None else args.seed + 1)
            if args.code_b
            else None
        )
        hgp = construct_hgp(code_a, code_b, args.field)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    spec_b = args.code_b or args.code_a
    seed_a = summarize_classical(hgp.code_a, args.code_a, compute_distance=compute_distance)
    seed_b = summarize_classical(hgp.code_b, spec_b, compute_distance=compute_distance)
    metadata = report(
        hgp,
        seed_a,
        seed_b,
        compute_distance=compute_distance,
        bound=args.bound,
    )
    metadata["spec_a"] = args.code_a
    metadata["spec_b"] = args.code_b or args.code_a

    if args.save_dir is not None:
        save_code(hgp, args.save_dir, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
