#!/usr/bin/env python3
"""Screen extxyz frames with fixed-composition chemistry and periodic Li graphs.

The ranking is a normalized Li graph geometry proxy. It does not estimate ionic
conductivity, migration barriers, or thermodynamic stability. Li sites joined
within ``r_cut`` form a periodic graph; the rank of its cycle image translations
separates finite clusters (0) from periodic 1D, 2D, and 3D components.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from glob import iglob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import iread as ase_iread
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

try:
    from smact import Element as SmactElement
    from smact.screening import pauling_test as smact_pauling_test
    SMACT_IMPORT_ERROR = ""
except Exception as exc:
    SmactElement = None
    smact_pauling_test = None
    SMACT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

HALS = ("F", "Cl", "Br", "I")
# Explicit ionic screening assumptions for the target oxyhalide chemistry.
# Nb permits mixtures of these states; these are hypotheses, not measured valences.
IONIC_OXIDATION_STATES = {"Li": (1,), "Nb": (3, 4, 5), "O": (-2,), **{s: (-1,) for s in HALS}}
SCORE_KIND = "li_periodic_geometry_proxy_v1"
GEOMETRY_COLUMNS = ("li_conn", "min_li_li", "li_percolation_dim", "li_percolation_fraction", "li_channel_score", "li_component_count", "li_periodic_components")
OUTPUT_COLUMNS = ["path", "frame", "dir", "spg", "formula", "n_atoms", "density", "f_li", "f_o", "hal_entropy", *GEOMETRY_COLUMNS, "passes_light_oxy", "charge_balance_ok", "charge_balance_status", "charge_balance_reason", "oxidation_state_guesses", "oxidation_state_policy", "smact_ok", "smact_status", "smact_reason", "quick_score", "score_kind", "filtered_reasons"]


@dataclass(frozen=True)
class ChemicalCheck:
    """An uncertain or failed check is never truthy success."""
    status: str
    reason: str = ""
    guesses: tuple[dict, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def iter_frames(path):
    yield from ase_iread(path, index=":")


def structure_from_atoms(atoms):
    if not bool(np.all(atoms.pbc)):
        raise ValueError("periodic screening requires PBC in all three lattice directions")
    return AseAtomsAdaptor.get_structure(atoms)


def comp_features(struct):
    el = struct.composition.get_el_amt_dict()
    n_li, n_o = el.get("Li", 0.0), el.get("O", 0.0)
    n_hal, n_tot = sum(el.get(x, 0.0) for x in HALS), sum(el.values())
    p = [el.get(x, 0.0) / n_hal for x in HALS if el.get(x, 0.0) > 0] if n_hal else []
    return {"formula": struct.composition.reduced_formula, "n_atoms": struct.num_sites,
            "density": float(struct.density), "f_li": n_li / n_tot if n_tot else 0.0,
            "f_o": n_o / (n_o + n_hal) if n_o + n_hal else 0.0,
            "hal_entropy": -sum(x * math.log(x) for x in p)}


def _integer_composition(comp: Composition, max_sites=128) -> Composition:
    """Preserve fixed stoichiometry; reduce exact rational counts before guessing."""
    fractions = {}
    for symbol, amount in comp.get_el_amt_dict().items():
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("composition amounts must be finite and positive")
        value = Fraction(str(amount)).limit_denominator(10000)
        if not math.isclose(float(value), amount, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("composition is not representable with bounded integer stoichiometry")
        fractions[symbol] = value
    if not fractions:
        raise ValueError("empty composition")
    denominator = math.lcm(*(v.denominator for v in fractions.values()))
    counts = {s: int(v * denominator) for s, v in fractions.items()}
    divisor = reduce(math.gcd, counts.values())
    counts = {s: n // divisor for s, n in counts.items()}
    if sum(counts.values()) > max_sites:
        raise ValueError(f"reduced formula exceeds chemical enumeration limit of {max_sites} sites")
    return Composition(counts)


def oxidation_state_policy(comp, overrides=None):
    """Use explicit oxyhalide states and pymatgen common states for other systems."""
    policy = {}
    for el in comp.elements:
        values = (overrides or {}).get(el.symbol, IONIC_OXIDATION_STATES.get(el.symbol, el.common_oxidation_states))
        policy[el.symbol] = tuple(sorted(set(int(v) for v in values)))
    return policy


def charge_balance_result(comp, oxidation_states=None, max_sites=128):
    try:
        fixed = _integer_composition(comp, max_sites)
        policy = oxidation_state_policy(fixed, oxidation_states)
        missing = [s for s, states in policy.items() if not states]
        if missing:
            return ChemicalCheck("unknown", "no_common_oxidation_states:" + ",".join(missing))
    except ValueError as exc:
        return ChemicalCheck("unknown", str(exc))
    except Exception as exc:
        return ChemicalCheck("error", f"{type(exc).__name__}: {exc}")
    try:
        # all_oxi_states=False plus an explicit override avoids silently broadening
        # chemistry. pymatgen permits mixtures within the supplied states.
        guesses = fixed.oxi_state_guesses(oxi_states_override=policy, all_oxi_states=False)
        finite = tuple(dict(g) for g in guesses if all(math.isfinite(float(v)) for v in g.values()))
        if len(finite) != len(guesses):
            return ChemicalCheck("error", "non_finite_oxidation_guess")
        return ChemicalCheck("pass", "fixed_stoichiometry_neutral", finite) if finite else ChemicalCheck("fail", "no_neutral_assignment_in_policy")
    except Exception as exc:
        return ChemicalCheck("error", f"{type(exc).__name__}: {exc}")


def charge_balance_ok(comp: Composition, oxidation_states=None) -> bool:
    return charge_balance_result(comp, oxidation_states).ok


def _valence_distribution(states, count, target):
    """Recover integer per-ion states from a mixed-valence average, using DP."""
    reachable = {0: ()}
    for _ in range(count):
        nxt = {}
        for subtotal, assignment in reachable.items():
            for valence in states:
                nxt.setdefault(subtotal + valence, assignment + (valence,))
        reachable = nxt
    return reachable.get(target)


def smact_result(comp, oxidation_states=None, charge_check=None, max_sites=128):
    if smact_pauling_test is None or SmactElement is None:
        return ChemicalCheck("unknown", "smact_unavailable:" + SMACT_IMPORT_ERROR)
    charge = charge_check or charge_balance_result(comp, oxidation_states, max_sites)
    if not charge.ok:
        return ChemicalCheck(charge.status, "charge_balance:" + charge.reason)
    try:
        fixed = _integer_composition(comp, max_sites)
        policy = oxidation_state_policy(fixed, oxidation_states)
        elements = {s: SmactElement(s) for s in fixed.get_el_amt_dict()}
        if any(e.pauling_eneg is None or not math.isfinite(float(e.pauling_eneg)) for e in elements.values()):
            return ChemicalCheck("unknown", "missing_pauling_electronegativity")
        for guess in charge.guesses:
            states, enegs, symbols = [], [], []
            for symbol, amount in fixed.get_el_amt_dict().items():
                total = float(guess[symbol]) * int(amount)
                if not math.isclose(total, round(total), abs_tol=1e-7):
                    return ChemicalCheck("error", "non_integer_mixed_valence_total")
                assignment = _valence_distribution(policy[symbol], int(amount), round(total))
                if assignment is None:
                    return ChemicalCheck("error", "cannot_recover_integer_valence_assignment")
                states.extend(assignment)
                enegs.extend([float(elements[symbol].pauling_eneg)] * int(amount))
                symbols.extend([symbol] * int(amount))
            # Real SMACT signature: oxidation states paired with corresponding
            # electronegativities. Counts here are the exact fixed formula.
            if bool(smact_pauling_test(states, enegs, symbols=symbols)):
                return ChemicalCheck("pass", "fixed_stoichiometry_pauling_pass", (guess,))
        return ChemicalCheck("fail", "pauling_electronegativity_fail")
    except Exception as exc:
        return ChemicalCheck("error", f"{type(exc).__name__}: {exc}")


def smact_ok(comp: Composition, oxidation_states=None) -> bool:
    return smact_result(comp, oxidation_states).ok


def _integer_rank(vectors):
    """Exact rank in Z^3; no floating-point tolerance for winding vectors."""
    nonzero = [tuple(int(x) for x in v) for v in vectors if any(v)]
    if not nonzero:
        return 0
    a = nonzero[0]
    cross = lambda x, y: (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
    normal = next((cross(a, b) for b in nonzero[1:] if any(cross(a, b))), None)
    if normal is None:
        return 1
    return 3 if any(sum(x*y for x, y in zip(normal, v)) for v in nonzero) else 2


def li_connectivity(struct, r_cut=3.0, super=(2, 2, 2)):
    """Li periodic graph descriptors, invariant under supercell representation.

    ``super`` is retained for CLI/API compatibility; exact periodic image edges
    replace finite supercell replication. ``li_conn`` now means the fraction of
    Li sites in components with nonzero periodic winding rank.
    """
    if not math.isfinite(r_cut) or r_cut <= 0:
        raise ValueError("r_cut must be finite and positive")
    if not struct.is_ordered:
        raise ValueError("Li graph screening requires an ordered structure")
    li_sites = [s for s in struct if s.specie.symbol == "Li"]
    n = len(li_sites)
    if not n:
        return {"li_conn": 0.0, "min_li_li": float("nan"), "li_percolation_dim": 0,
                "li_percolation_fraction": 0.0, "li_channel_score": 0.0,
                "li_component_count": 0, "li_periodic_components": 0}
    # A Li-only structure gives stable local indices for pymatgen image edges.
    li = Structure(struct.lattice, ["Li"] * n, [s.frac_coords for s in li_sites])
    centers, neighbors, images, distances = li.get_neighbor_list(r_cut)
    graph = [[] for _ in range(n)]
    for u, v, image, distance in zip(centers, neighbors, images, distances):
        if distance > r_cut + 1e-8:
            continue
        shift = tuple(int(round(x)) for x in image)
        if int(u) == int(v) and not any(shift):
            continue
        graph[int(u)].append((int(v), shift))
        graph[int(v)].append((int(u), tuple(-x for x in shift)))
    unseen = set(range(n))
    components = []
    while unseen:
        start = min(unseen)
        potentials = {start: (0, 0, 0)}
        queue, windings = deque([start]), []
        unseen.remove(start)
        while queue:
            u = queue.popleft()
            for v, image in graph[u]:
                potential = tuple(a + b for a, b in zip(potentials[u], image))
                if v not in potentials:
                    potentials[v] = potential
                    unseen.discard(v)
                    queue.append(v)
                else:
                    windings.append(tuple(a - b for a, b in zip(potential, potentials[v])))
        components.append((len(potentials), _integer_rank(windings)))
    periodic = sum(size for size, rank in components if rank > 0)
    # A nonzero lattice image of each Li exists inside min(lattice.abc); this
    # also includes any shorter intersite pair. No sentinel distance is used.
    nearest = li.get_neighbor_list(min(li.lattice.abc) + 1e-7)[3]
    positive = nearest[nearest > 1e-8]
    min_distance = float(np.min(positive)) if positive.size else float("nan")
    return {"li_conn": periodic / n, "min_li_li": min_distance,
            "li_percolation_dim": max(rank for _, rank in components),
            "li_percolation_fraction": periodic / n,
            "li_channel_score": sum(size * rank for size, rank in components) / (3 * n),
            "li_component_count": len(components),
            "li_periodic_components": sum(rank > 0 for _, rank in components)}


def quick_score_row(row):
    """0..1 normalized periodic Li geometry, without empirical distance bonuses."""
    if row.get("li_percolation_dim", 0) == 0:
        return 0.0
    score = row.get("li_channel_score", 0.0)
    return max(0.0, min(1.0, float(score))) if score is not None and math.isfinite(float(score)) else 0.0


def guess_spg(struct):
    try:
        return SpacegroupAnalyzer(struct, symprec=0.2, angle_tolerance=5).get_space_group_symbol()
    except Exception:
        return "unknown"


def _parse_oxidation_states(values):
    overrides = {}
    for value in values or []:
        symbol, sep, states = value.partition("=")
        if not sep or not Element.is_valid_symbol(symbol):
            raise ValueError("oxidation states must have the form Nb=3,4,5")
        parsed = tuple(int(v) for v in states.split(","))
        if not parsed:
            raise ValueError("oxidation-state list may not be empty")
        overrides[symbol] = parsed
    return overrides


def _check_structure(struct):
    if not struct.is_ordered or not struct.num_sites:
        raise ValueError("structure must be ordered and nonempty")
    if not np.isfinite(struct.lattice.matrix).all() or not np.isfinite(struct.frac_coords).all():
        raise ValueError("structure contains non-finite lattice or positions")
    if not math.isfinite(struct.volume) or struct.volume <= 0:
        raise ValueError("structure volume must be finite and positive")
    # Coincident sites are invalid graph inputs and cannot be scored safely.
    if struct.num_sites > 1:
        distances = struct.distance_matrix
        np.fill_diagonal(distances, float("inf"))
        if np.min(distances) <= 1e-8:
            raise ValueError("structure contains coincident sites")


def screen_structure(struct, *, r_cut=3.0, required_elements=("Li",), allowed_elements=None, light_oxy=(0.05, 0.35), require_charge_balance=True, use_smact=True, oxidation_states=None, max_chem_sites=128, chemistry_cache=None):
    """Return a complete descriptor/audit row; rejected frames retain reasons."""
    _check_structure(struct)
    features = comp_features(struct)
    symbols = struct.composition.get_el_amt_dict()
    reasons = []
    if symbols.get("Li", 0) <= 0:
        reasons.append("missing_li")
    missing = sorted(set(required_elements) - symbols.keys())
    if missing:
        reasons.append("missing_required_elements:" + ",".join(missing))
    unexpected = sorted(symbols.keys() - set(allowed_elements)) if allowed_elements is not None else []
    if unexpected:
        reasons.append("unexpected_elements:" + ",".join(unexpected))
    passes_oxy = light_oxy is None or light_oxy[0] <= features["f_o"] <= light_oxy[1]
    if not passes_oxy:
        reasons.append("light_oxy_out_of_range")
    key = (tuple(sorted(symbols.items())), tuple(sorted((s, tuple(v)) for s, v in (oxidation_states or {}).items())), max_chem_sites, require_charge_balance, use_smact)
    cached = chemistry_cache.get(key) if chemistry_cache is not None else None
    if cached is None:
        charge = charge_balance_result(struct.composition, oxidation_states, max_chem_sites) if (require_charge_balance or use_smact) else ChemicalCheck("not_requested")
        smact = smact_result(struct.composition, oxidation_states, charge, max_chem_sites) if use_smact else ChemicalCheck("not_requested")
        cached = (charge, smact)
        if chemistry_cache is not None:
            chemistry_cache[key] = cached
    charge, smact = cached
    if require_charge_balance and not charge.ok:
        reasons.append("charge_balance_" + charge.status)
    if use_smact and not smact.ok:
        reasons.append("smact_" + smact.status)
    geometry = li_connectivity(struct, r_cut=r_cut)
    row = {"spg": guess_spg(struct), **features, **geometry,
           "passes_light_oxy": passes_oxy,
           "charge_balance_ok": charge.ok if charge.status != "not_requested" else None,
           "charge_balance_status": charge.status, "charge_balance_reason": charge.reason,
           "oxidation_state_guesses": json.dumps(charge.guesses, sort_keys=True),
           "oxidation_state_policy": json.dumps(oxidation_state_policy(struct.composition, oxidation_states), sort_keys=True),
           "smact_ok": smact.ok if smact.status != "not_requested" else None,
           "smact_status": smact.status, "smact_reason": smact.reason,
           "score_kind": SCORE_KIND}
    row["quick_score"] = quick_score_row(row)
    finite_columns = ["density", "f_li", "f_o", "hal_entropy", *GEOMETRY_COLUMNS, "quick_score"]
    if any(not math.isfinite(float(row[k])) for k in finite_columns):
        reasons.append("non_finite_descriptor")
    row["filtered_reasons"] = ";".join(reasons)
    return row


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=".", help="执行目录（所有相对路径均相对该目录）")
    ap.add_argument("--base", default="results/chemical_system_energy_above_hull", help="递归查找 relaxed.extxyz 的根目录")
    ap.add_argument("--out", default="results/stage2_candidates.csv", help="接受样品 CSV")
    ap.add_argument("--screened-out", default="screened_out.csv", help="拒绝/错误审计 CSV（总会写出）")
    ap.add_argument("--r-cut", type=float, default=3.0, help="Li 周期图邻接截断 (Å)，仅为几何代理")
    ap.add_argument("--super", type=int, nargs=3, default=(2, 2, 2), help="兼容参数；周期 image 图不需要复制超胞")
    ap.add_argument("--required-elements", nargs="+", default=["Li"], help="必须出现的元素；目标体系可用 Li Nb O Cl（默认 Li）")
    ap.add_argument("--allowed-elements", nargs="+", default=None, help="可选元素白名单，锁定目标体系时可用 Li Nb O Cl")
    oxy = ap.add_mutually_exclusive_group()
    oxy.add_argument("--light-oxy", type=float, nargs=2, default=(0.05, 0.35), help="实际过滤 f_o=O/(O+卤) 的区间（默认 0.05 0.35）")
    oxy.add_argument("--no-light-oxy", action="store_true", help="显式关闭氧比例过滤，适用于其他化学体系")
    charge = ap.add_mutually_exclusive_group()
    charge.add_argument("--require-charge-balance", dest="require_charge_balance", action="store_true", help="固定计量式电中性校验（默认开启）")
    charge.add_argument("--no-charge-balance", dest="require_charge_balance", action="store_false", help="显式关闭独立电中性过滤；SMACT自身仍校验电中性")
    smact = ap.add_mutually_exclusive_group()
    smact.add_argument("--use-smact", dest="use_smact", action="store_true", help="真实 SMACT Pauling 规则校验（默认开启）")
    smact.add_argument("--no-smact", dest="use_smact", action="store_false", help="显式关闭 SMACT")
    ap.set_defaults(require_charge_balance=True, use_smact=True)
    ap.add_argument("--oxidation-states", action="append", default=[], metavar="ELEMENT=VALENCES", help="覆盖价态假设，例如 --oxidation-states Nb=3,4,5")
    ap.add_argument("--max-chem-sites", type=int, default=128, help="约简计量式的价态枚举上限；超限以 unknown 拒绝")
    ap.add_argument("--topk", type=int, default=150, help="几何代理排序引用数量，未经 DFT/MD 验证")
    ap.add_argument("--refs-out", default="top150_refs.txt", help="几何排序引用列表，格式 path::frame")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not math.isfinite(args.r_cut) or args.r_cut <= 0 or args.topk < 0 or args.max_chem_sites < 1:
        ap.error("r-cut/max-chem-sites must be positive, and topk nonnegative")
    if any(n < 1 for n in args.super):
        ap.error("super dimensions must be positive")
    if not all(Element.is_valid_symbol(s) for s in args.required_elements):
        ap.error("required-elements contains an invalid chemical symbol")
    if args.allowed_elements is not None:
        if not all(Element.is_valid_symbol(s) for s in args.allowed_elements):
            ap.error("allowed-elements contains an invalid chemical symbol")
        if not set(args.required_elements).issubset(args.allowed_elements) or "Li" not in args.allowed_elements:
            ap.error("allowed-elements must include Li and every required element")
    if not args.no_light_oxy and (not all(math.isfinite(v) for v in args.light_oxy) or not 0 <= args.light_oxy[0] <= args.light_oxy[1] <= 1):
        ap.error("light-oxy requires 0 <= lower <= upper <= 1")
    try:
        overrides = _parse_oxidation_states(args.oxidation_states)
    except ValueError as exc:
        ap.error(str(exc))
    if args.use_smact and (smact_pauling_test is None or SmactElement is None):
        ap.error("SMACT is required by default but unavailable; install smact or explicitly use --no-smact. " + SMACT_IMPORT_ERROR)
    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.is_dir():
        ap.error("workdir does not exist")
    # Resolve paths without changing process CWD, so callers remain unaffected.
    resolve = lambda p: Path(p) if Path(p).is_absolute() else workdir / p
    out, rejected_out, refs_out = map(resolve, (args.out, args.screened_out, args.refs_out))
    for path in (out, rejected_out, refs_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    rows, rejected, cache = [], [], {}
    files = sorted(iglob(str(resolve(args.base) / "**" / "relaxed.extxyz"), recursive=True))
    for path in tqdm(files, desc="Scanning files"):
        try:
            for index, atoms in enumerate(iter_frames(path)):
                identity = {"path": path, "frame": index, "dir": os.path.dirname(path)}
                try:
                    row = {**identity, **screen_structure(structure_from_atoms(atoms), r_cut=args.r_cut, required_elements=args.required_elements, allowed_elements=args.allowed_elements, light_oxy=None if args.no_light_oxy else args.light_oxy, require_charge_balance=args.require_charge_balance, use_smact=args.use_smact, oxidation_states=overrides, max_chem_sites=args.max_chem_sites, chemistry_cache=cache)}
                    (rejected if row["filtered_reasons"] else rows).append(row)
                except Exception as exc:
                    rejected.append({**identity, "filtered_reasons": "structure_error", "charge_balance_status": "error", "charge_balance_reason": f"{type(exc).__name__}: {exc}", "score_kind": SCORE_KIND})
        except Exception as exc:
            rejected.append({"path": path, "dir": os.path.dirname(path), "filtered_reasons": "file_read_error", "charge_balance_status": "error", "charge_balance_reason": f"{type(exc).__name__}: {exc}", "score_kind": SCORE_KIND})
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not df.empty:
        df.sort_values(["quick_score", "path", "frame"], ascending=[False, True, True], kind="stable", inplace=True)
    df.to_csv(out, index=False)
    pd.DataFrame(rejected, columns=OUTPUT_COLUMNS).to_csv(rejected_out, index=False)
    refs = [f"{r.path}::{int(r.frame)}" for r in df.head(args.topk).itertuples()]
    refs_out.write_text("\n".join(refs), encoding="utf-8")
    print(f"完成: 接受 {len(df)} 帧，拒绝/错误 {len(rejected)} 条 -> {out}")
    print(f"拒绝审计: {rejected_out}; 几何代理引用: {refs_out}")
    print("quick_score 为归一化周期 Li 图几何代理，未估计电导率、迁移势垒或稳定性。")
    if not files:
        print(f"[WARN] 未在 {resolve(args.base)} 下找到 relaxed.extxyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
