#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

批量为 relaxed.extxyz 中的所有候选计算 formation_energy_per_atom 与 E_above_hull。



两种模式：

  1) --mode mp      : 适用于你已有 **DFT 总能** 的情况。你的候选条目会套用

                      MaterialsProject2020Compatibility 校正，并与 MP 的 get_entries_in_chemsys

                      返回的竞争相一起建相图（同一修正，基线可比）。

  2) --mode chgnet  : 候选与 MP 竞争相统一经 MatterSim 晶胞/原子弛豫后，用 CHGNet 单点预测能量，

                      在同一 ML 基线上构造“模型凸包”。适合前期大批量筛选。



用法示例：

  MP/DFT 基线（建议先把每帧的 DFT 能量放在 CSV 里）：

    python workflow/tools/compute_hull_from_relaxed.py \

      --root results/_segments \

      --mode mp --energies-csv dft_energies.csv --out results/hull_mp.csv



  纯 CHGNet 基线（无需 DFT 能量；会为每个 chemsys 拉取 MP 结构并用 CHGNet 取能）：

    python workflow/tools/compute_hull_from_relaxed.py --root results/_segments --mode chgnet --max-mp-competitors 300 --out results/hull_chgnet.csv



注意：

- 访问 Materials Project 需设置环境变量 MP_API_KEY，或用 --mp-api-key 传入。

- MP 模式下，若混入非同修正/非同功能的能量会产生系统误差；建议按 MP2020 方案做同样校正。

"""



from __future__ import annotations

import argparse
import hashlib
import math
import sys
import tempfile
import numpy as np

import csv

import json

import os

from pathlib import Path

from collections import defaultdict
from itertools import combinations
from time import sleep

from typing import Dict, List, Tuple, Optional



from ase.io import iread

from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.core import Structure

from pymatgen.entries.computed_entries import ComputedStructureEntry

from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from pymatgen.analysis.phase_diagram import PhaseDiagram

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from mattersim_relaxation import (MatterSimRelaxer, add_relaxation_arguments,
                                 settings_from_args, save_relaxed_structure)
ENERGY_MODEL = "CHGNet-0.3.0"



def parse_args(argv=None):

    p = argparse.ArgumentParser()

    p.add_argument("--root", required=True,

                   help="递归搜索该目录下的 relaxed.extxyz")

    p.add_argument("--out", required=True, help="输出 CSV 路径")

    p.add_argument("--mode", choices=["mp", "chgnet"], required=True,

                   help="mp: DFT+MP 校正的正式凸包；chgnet: 纯 ML 模型凸包")

    p.add_argument("--energies-csv", default=None,

                   help="仅用于 --mode mp。CSV 至少包含两列：id,energy_eV；"

                        "id 形如 <extxyz绝对路径>#<frame_index>。")

    p.add_argument("--mp-api-key", default=os.environ.get("MP_API_KEY"),

                   help="Materials Project API Key（默认取环境变量 MP_API_KEY）")

    p.add_argument("--max-mp-competitors", type=int, default=None,

                   help="仅用于 --mode chgnet。每个化学体系最多采样的 MP 竞争相数量（默认不限）。")

    add_relaxation_arguments(p)
    args = p.parse_args(argv)
    try:
        settings_from_args(args)
    except ValueError as exc:
        p.error(str(exc))
    return args





def get_chemsys(struct: Structure) -> str:

    return "-".join(sorted({el.symbol for el in struct.composition.elements}))





def load_relaxed_frames(root: Path) -> List[Tuple[str, int, Structure]]:

    """扫描 root 下所有 relaxed.extxyz，返回 [(extxyz_path, frame_idx, pmg_structure), ...]"""

    frames = []

    adaptor = AseAtomsAdaptor()

    for ext in root.rglob("relaxed.extxyz"):

        try:

            for i, atoms in enumerate(iread(str(ext))):

                struct = adaptor.get_structure(atoms)

                frames.append((str(ext.resolve()), i, struct))

        except Exception as e:

            print(f"[WARN] 读取 {ext} 失败：{e}")

    return frames





def read_energy_map_from_csv(csv_path: Path) -> Dict[str, float]:

    """读取 id -> energy_eV 映射。id 形如 '/abs/path/relaxed.extxyz#12'。"""

    mp: Dict[str, float] = {}

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        if "id" not in reader.fieldnames or "energy_eV" not in reader.fieldnames:

            raise ValueError("energies-csv 需要包含列: id, energy_eV")

        for row in reader:

            mp[row["id"]] = float(row["energy_eV"])

    return mp





def build_entries_mp(frames, energy_map) -> Dict[str, List[ComputedStructureEntry]]:

    """

    构造 MP/DFT 基线下的候选 entries，并按化学体系分组。

    要求 energy_map 提供每一帧的 DFT 总能（eV/晶胞）。

    """

    compat = MaterialsProject2020Compatibility()  # 与 MP 同一校正方案

    groups = defaultdict(list)

    for ext_path, idx, struct in frames:

        uid = f"{ext_path}#{idx}"

        if uid not in energy_map:

            print(f"[WARN] 缺少能量：{uid}，跳过。")

            continue

        e = energy_map[uid]

        ce = ComputedStructureEntry(structure=struct, energy=e)

        ce = compat.process_entry(ce)  # 应用 MP2020 校正

        cs = get_chemsys(struct)

        groups[cs].append(ce)

    return groups





def chgnet_energy_for_structure(struct: Structure, calc: CHGNetCalculator) -> float:

    """用 CHGNetCalculator 对给定结构做单点能量（eV/晶胞）。"""

    atoms = AseAtomsAdaptor.get_atoms(struct)

    atoms.calc = calc

    e = float(atoms.get_potential_energy())
    if not math.isfinite(e):
        raise ValueError("CHGNet produced a non-finite single-point energy")
    return e





def _chgnet_calculator(device=None):
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    model = CHGNet.load(model_name="0.3.0", use_device=device)
    return CHGNetCalculator(model=model, use_device=device)


def _structure_source_id(struct):
    source_id = getattr(struct, "properties", {}).get("source_id")
    if source_id:
        return str(source_id)
    fingerprint = json.dumps({"composition": struct.composition.get_el_amt_dict(),
                              "lattice": struct.lattice.matrix.tolist(),
                              "frac_coords": struct.frac_coords.tolist()}, sort_keys=True)
    return "reference:" + hashlib.sha256(fingerprint.encode()).hexdigest()


def _trace_paths(output_dir, kind, source_id):
    if output_dir is None:
        return None, None
    folder = Path(output_dir) / kind
    folder.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(source_id.encode()).hexdigest()[:24]
    return folder / (stem + ".cif"), folder / (stem + ".audit.json")


def _write_trace(path, trace):
    if path is None:
        return
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(trace, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _verify_relaxed(original, relaxed, audit, settings):
    if audit.get("status") != "converged" or audit.get("converged") is not True:
        raise ValueError("MatterSim relaxation did not return a verified converged audit")
    if audit.get("settings") != settings:
        raise ValueError("MatterSim relaxation settings differ from the shared settings")
    if not isinstance(relaxed, Structure) or not relaxed.is_ordered or not relaxed.num_sites:
        raise ValueError("MatterSim returned an invalid structure")
    if relaxed.composition.get_el_amt_dict() != original.composition.get_el_amt_dict():
        raise ValueError("MatterSim relaxation changed composition")
    if not np.isfinite(relaxed.lattice.matrix).all() or not np.isfinite(relaxed.cart_coords).all() or not math.isfinite(relaxed.volume) or relaxed.volume <= 0:
        raise ValueError("MatterSim returned non-finite or invalid geometry")
    if not math.isfinite(float(audit.get("fmax_final", math.nan))) or audit["fmax_final"] > settings["fmax"] * (1 + 1e-7):
        raise ValueError("MatterSim final forces do not satisfy the configured threshold")


def _failure_trace(trace, exc):
    trace.update(status="calculation_failed", error=f"{type(exc).__name__}: {exc}")
    if hasattr(exc, "audit"):
        trace["relaxation_audit"] = exc.audit
    return trace


def build_entries_chgnet(frames, device: Optional[str] = None, relaxer=None,
                         output_dir=None) -> Dict[str, List[ComputedStructureEntry]]:
    """MatterSim-relax every candidate, then use CHGNet single-point energy.

    Candidate failures are explicitly audited and excluded. The same relaxer
    must also be supplied to ``entries_from_structures_with_chgnet`` for MP.
    """
    relaxer = relaxer if relaxer is not None else MatterSimRelaxer(device=device)
    settings = relaxer.settings.as_dict()
    calc = None
    groups = defaultdict(list)
    for ext_path, idx, struct in frames:
        uid = f"{ext_path}#{idx}"
        cif_path, audit_path = _trace_paths(output_dir, "candidates", uid)
        trace = {"source_id": uid, "source_path": str(ext_path), "frame": idx,
                 "path": None, "settings": settings, "status": "calculation_failed",
                 "energy_model": ENERGY_MODEL,
                 "formula": struct.composition.reduced_formula, "chemsys": get_chemsys(struct)}
        try:
            relaxed, audit = relaxer.relax(struct)
            trace["relaxation_audit"] = audit
            _verify_relaxed(struct, relaxed, audit, settings)
            if cif_path is not None:
                save_relaxed_structure(relaxed, cif_path)
                trace["path"] = str(cif_path.resolve())
            if calc is None:
                calc = _chgnet_calculator(device)
            energy = chgnet_energy_for_structure(relaxed, calc)
            if not math.isfinite(energy):
                raise ValueError("CHGNet produced a non-finite single-point energy")
            trace.update(status="complete", energy_total_eV=energy)
            entry = ComputedStructureEntry(relaxed, energy, entry_id=uid, data=trace.copy())
            _write_trace(audit_path, trace)
            groups[get_chemsys(relaxed)].append(entry)
        except Exception as exc:
            _write_trace(audit_path, _failure_trace(trace, exc))
            print(f"[WARN] MatterSim/CHGNet candidate failed and was excluded: {uid} -> {exc}")
    return groups


def fetch_mp_competitors_entries(chemsys: str, mpr: MPRester) -> List:

    """

    取 MP thermo entries（已在服务器端做过统一修正/混合），用于 mp 模式。

    """

    entries = mpr.get_entries_in_chemsys(chemsys)

    return entries








def _mp_reference_structures(docs):
    structures = []
    for doc in docs:
        struct = doc.structure.copy()
        struct.properties["source_id"] = str(doc.material_id)
        structures.append(struct)
    return structures


def fetch_mp_competitors_structures(chemsys: str, mpr, limit: Optional[int] = None) -> List[Structure]:
    """
    分批抓取 MP 竞争相：对 elems 的所有子集，用 chemsys 精确查询。
    优先稳定相，并确保补齐每个元素的端元（unary）。
    """
    from mp_api.client.core.client import MPRestError

    elems = chemsys.split("-")
    out_docs = []
    seen_mids = set()

    def search_safe(**kwargs):
        # 轻量重试，缓解临时网络/限流
        for i in range(3):
            try:
                return list(mpr.materials.summary.search(**kwargs))
            except MPRestError:
                sleep(1.5 * (i + 1))
        # 最后一次再抛出（让上游看见真实错误）
        return list(mpr.materials.summary.search(**kwargs))

    # 1) 先确保每个端元都有（unary）
    for el in elems:
        docs_u = search_safe(chemsys=el, fields=["structure", "material_id", "is_stable", "nelements"])
        pick = next((d for d in docs_u if getattr(d, "is_stable", False)), docs_u[0] if docs_u else None)
        if pick and pick.material_id not in seen_mids:
            out_docs.append(pick); seen_mids.add(pick.material_id)

    # 2) 再从低阶到高阶子体系累积（binary/ternary/...）
    max_k = len(elems)
    for k in range(2, max_k + 1):
        for comb in combinations(elems, k):
            cs = "-".join(sorted(comb))
            docs = search_safe(chemsys=cs, fields=["structure", "material_id", "is_stable", "nelements"])
            # 稳定相优先；不稳定相作为补充
            docs_stable = [d for d in docs if getattr(d, "is_stable", False)]
            docs_other  = [d for d in docs if not getattr(d, "is_stable", False)]
            for d in (docs_stable + docs_other):
                if d.material_id in seen_mids:
                    continue
                out_docs.append(d); seen_mids.add(d.material_id)
                if limit is not None and len(out_docs) >= limit:
                    return _mp_reference_structures(out_docs)

    return _mp_reference_structures(out_docs)





def entries_from_structures_with_chgnet(structs: List[Structure], device: Optional[str] = None,
                                      relaxer=None, output_dir=None) -> List[ComputedStructureEntry]:
    """Uniformly relax/score every reference; any failure aborts the phase set."""
    relaxer = relaxer if relaxer is not None else MatterSimRelaxer(device=device)
    settings = relaxer.settings.as_dict()
    calc = None
    entries = []
    for struct in structs:
        source_id = _structure_source_id(struct)
        cif_path, audit_path = _trace_paths(output_dir, "references", source_id)
        trace = {"source_id": source_id, "source_path": source_id, "path": None,
                 "settings": settings, "status": "calculation_failed",
                 "energy_model": ENERGY_MODEL,
                 "formula": struct.composition.reduced_formula, "chemsys": get_chemsys(struct)}
        try:
            relaxed, audit = relaxer.relax(struct)
            trace["relaxation_audit"] = audit
            _verify_relaxed(struct, relaxed, audit, settings)
            if cif_path is not None:
                save_relaxed_structure(relaxed, cif_path)
                trace["path"] = str(cif_path.resolve())
            if calc is None:
                calc = _chgnet_calculator(device)
            energy = chgnet_energy_for_structure(relaxed, calc)
            if not math.isfinite(energy):
                raise ValueError("CHGNet produced a non-finite single-point energy")
            trace.update(status="complete", energy_total_eV=energy)
            entry = ComputedStructureEntry(relaxed, energy, entry_id=source_id, data=trace.copy())
            _write_trace(audit_path, trace)
            entries.append(entry)
        except Exception as exc:
            _write_trace(audit_path, _failure_trace(trace, exc))
            raise RuntimeError(f"Competing MP reference failed MatterSim/CHGNet; phase diagram aborted: {source_id}") from exc
    return entries


def _entry_provenance(entry):
    data = entry.data or {}
    audit = data.get("relaxation_audit", {})
    return {"source_id": data.get("source_id", entry.entry_id or ""),
            "source_path": data.get("source_path", ""), "path": data.get("path", ""),
            "relaxation_settings": json.dumps(data.get("settings", {}), sort_keys=True),
            "relaxation_status": audit.get("status", "not_applicable"),
            "energy_model": data.get("energy_model", "MP/DFT"),
            "calculation_status": data.get("status", "complete"), "error": data.get("error", "")}


def _candidate_failure_rows(frames, output_dir):
    rows = []
    for ext_path, index, _ in frames:
        uid = f"{ext_path}#{index}"
        _, audit_path = _trace_paths(output_dir, "candidates", uid)
        if not audit_path.exists():
            continue
        trace = json.loads(audit_path.read_text())
        if trace.get("status") == "complete":
            continue
        rows.append({"chemsys": trace.get("chemsys", ""), "formula": trace.get("formula", ""),
                     "source_id": uid, "source_path": trace.get("source_path", ""),
                     "path": trace.get("path") or "",
                     "relaxation_settings": json.dumps(trace.get("settings", {}), sort_keys=True),
                     "relaxation_status": trace.get("relaxation_audit", {}).get("status", "failed"),
                     "energy_model": trace.get("energy_model", ENERGY_MODEL),
                     "calculation_status": "calculation_failed", "error": trace.get("error", "")})
    return rows


def main(argv=None):

    args = parse_args(argv)

    if args.mode == "chgnet":
        # A failed ML rerun must not leave an older successful hull CSV looking
        # current. The DFT branch retains its existing output semantics.
        Path(args.out).unlink(missing_ok=True)

    from mp_api.client import MPRester

    root = Path(args.root).resolve()

    frames = load_relaxed_frames(root)

    if not frames:

        print("[ERR] 未找到任何 relaxed.extxyz 帧。")

        return



    # 组装候选 entries
    relaxation_dir = Path(args.out).resolve().parent / "relaxation" / "all_frames"
    relaxer = None

    if args.mode == "mp":

        if not args.energies_csv:

            raise SystemExit("[ERR] --mode mp 需要提供 --energies-csv（DFT 总能，单位 eV/晶胞）。")

        energy_map = read_energy_map_from_csv(Path(args.energies_csv))

        grouped_entries = build_entries_mp(frames, energy_map)

    else:  # chgnet

        relaxer = MatterSimRelaxer(settings=settings_from_args(args))
        grouped_entries = build_entries_chgnet(frames, relaxer=relaxer, output_dir=relaxation_dir)



    # 逐化学体系构建相图并评估

    out_rows = _candidate_failure_rows(frames, relaxation_dir) if args.mode == "chgnet" else []

    with MPRester(api_key=args.mp_api_key) as mpr:

        for csys, my_entries in grouped_entries.items():

            if not my_entries:

                continue



            # 准备竞争相

            if args.mode == "mp":

                comp_entries = fetch_mp_competitors_entries(csys, mpr)  # 服务器端的 thermo entries（含修正）。

            else:

                comp_structs = fetch_mp_competitors_structures(csys, mpr, args.max_mp_competitors)

                comp_entries = entries_from_structures_with_chgnet(comp_structs, relaxer=relaxer, output_dir=relaxation_dir)

            # 假设变量名分别是：
            #   csys: 当前的 "Li-Cl-..." 字符串
            #   comp_entries: MP 竞争相（用上面函数取回后经 CHGNet 打分再包成 PDEntry）
            #   my_entries:   你自己生成/松弛后的候选条目
            all_elems = set(csys.split("-"))
            elems_in_comp = set()
            for e in (comp_entries or []):
                try:
                    elems_in_comp.update({el.symbol for el in e.composition.elements})
                except Exception:
                    pass
            missing = all_elems - elems_in_comp
            if missing:
                print(f"[WARN] 竞争相缺少端元 {[e for e in sorted(missing)]}，跳过 {csys}。")
                continue  # 直接跳过该体系，避免 PhaseDiagram 报错


            # 建相图并计算指标

            try:

                pd = PhaseDiagram(comp_entries + my_entries)  # 标准 PD；MP 网站也是用这套。

            except Exception as e:

                print(f"[WARN] PhaseDiagram 失败（{csys}）：{e}")

                continue



            for ce in my_entries:

                try:

                    e_above = float(pd.get_e_above_hull(ce))        # eV/atom

                    e_form_pa = float(pd.get_form_energy_per_atom(ce))

                    is_stable = (e_above <= 1e-6)

                    decomp = pd.get_decomposition(ce.composition)
                    parts = []
                    for k, amt in decomp.items():
                        # k 可能是 Entry（有 composition），也可能是 Composition/Formula-like
                        try:
                            rf = k.composition.reduced_formula  # Entry 路径
                        except Exception:
                            try:
                                rf = k.reduced_formula           # Composition 路径
                            except Exception:
                                rf = str(getattr(k, "formula", k))
                        parts.append(f"{amt:.3f}*{rf}")
                    decomp_str = " + ".join(parts)

                    out_rows.append({

                        **_entry_provenance(ce),

                        "chemsys": csys,

                        "formula": ce.composition.reduced_formula,

                        "energy_eV_cell": ce.energy,

                        "natoms_cell": ce.composition.num_atoms,

                        "energy_eV_per_atom": ce.energy_per_atom,

                        "formation_energy_per_atom": e_form_pa,

                        "energy_above_hull": e_above,

                        "is_stable": int(is_stable),

                        "decomposition": decomp_str

                    })

                except Exception as e:

                    print(f"[WARN] 指标计算失败（{csys}, {ce.composition.reduced_formula}）：{e}")



    # 写出 CSV

    os.makedirs(Path(args.out).parent, exist_ok=True)

    fieldnames = ["chemsys", "formula", "energy_eV_cell", "natoms_cell",

                  "energy_eV_per_atom", "formation_energy_per_atom",

                  "energy_above_hull", "is_stable", "decomposition",
                  "source_id", "source_path", "path", "relaxation_settings",
                  "relaxation_status", "calculation_status", "error"]
    fieldnames.append("energy_model")

    with open(args.out, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for r in out_rows:

            writer.writerow(r)



    print(f"[OK] 写出 {len(out_rows)} 条记录 -> {args.out}")





if __name__ == "__main__":

    main()
