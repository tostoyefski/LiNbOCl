#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

批量为 relaxed.extxyz 中的所有候选计算 formation_energy_per_atom 与 E_above_hull。



两种模式：

  1) --mode mp      : 适用于你已有 **DFT 总能** 的情况。你的候选条目会套用

                      MaterialsProject2020Compatibility 校正，并与 MP 的 get_entries_in_chemsys

                      返回的竞争相一起建相图（同一修正，基线可比）。

  2) --mode chgnet  : 全部用 CHGNet (ASE Calculator) 预测能量（你的候选 + MP 竞争相结构），

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



def parse_args():

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

    return p.parse_args()





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

    # 允许 CHGNet 自己构图；如出现孤立原子，ASE/CHGNet 会给出警告
    e = float(atoms.get_potential_energy())

    return e





def build_entries_chgnet(frames, device: Optional[str] = None) -> Dict[str, List[ComputedStructureEntry]]:

    """

    用 CHGNet 为候选结构取能量，并按化学体系分组（纯 ML 基线）。

    """

    from chgnet.model.dynamics import CHGNetCalculator

    calc = CHGNetCalculator() if device is None else CHGNetCalculator(use_device=device)

    groups = defaultdict(list)


    import math
    for ext_path, idx, struct in frames:
        try:
            e = chgnet_energy_for_structure(struct, calc)
            if not (isinstance(e, (int, float)) and math.isfinite(e)):
                print(f"[WARN] 非有限能量，跳过：{ext_path}#{idx}  -> {e}")
                continue
        except Exception as eerr:
            print(f"[WARN] CHGNet 取能失败：{ext_path}#{idx} -> {eerr}")
            continue
        ce = ComputedStructureEntry(structure=struct, energy=e)

        cs = get_chemsys(struct)

        groups[cs].append(ce)

    return groups





def fetch_mp_competitors_entries(chemsys: str, mpr: MPRester) -> List:

    """

    取 MP thermo entries（已在服务器端做过统一修正/混合），用于 mp 模式。

    """

    entries = mpr.get_entries_in_chemsys(chemsys)

    return entries








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
                    return [doc.structure for doc in out_docs]

    return [doc.structure for doc in out_docs]





def entries_from_structures_with_chgnet(structs: List[Structure], device: Optional[str] = None) -> List[ComputedStructureEntry]:

    from chgnet.model.dynamics import CHGNetCalculator

    calc = CHGNetCalculator() if device is None else CHGNetCalculator(use_device=device)

    entries = []

    import math
    for s in structs:

        try:

            e = chgnet_energy_for_structure(s, calc)

            if not (isinstance(e, (int,float)) and math.isfinite(e)):
                print("[WARN] 竞争相能量为非有限值，跳过")
                continue
            entries.append(ComputedStructureEntry(s, e))

        except Exception as eerr:

            print(f"[WARN] 竞争相 CHGNet 取能失败：{eerr}")

    return entries





def main():

    args = parse_args()

    from mp_api.client import MPRester

    root = Path(args.root).resolve()

    frames = load_relaxed_frames(root)

    if not frames:

        print("[ERR] 未找到任何 relaxed.extxyz 帧。")

        return



    # 组装候选 entries

    if args.mode == "mp":

        if not args.energies_csv:

            raise SystemExit("[ERR] --mode mp 需要提供 --energies-csv（DFT 总能，单位 eV/晶胞）。")

        energy_map = read_energy_map_from_csv(Path(args.energies_csv))

        grouped_entries = build_entries_mp(frames, energy_map)

    else:  # chgnet

        grouped_entries = build_entries_chgnet(frames)



    # 逐化学体系构建相图并评估

    out_rows = []

    with MPRester(api_key=args.mp_api_key) as mpr:

        for csys, my_entries in grouped_entries.items():

            if not my_entries:

                continue



            # 准备竞争相

            if args.mode == "mp":

                comp_entries = fetch_mp_competitors_entries(csys, mpr)  # 服务器端的 thermo entries（含修正）。

            else:

                comp_structs = fetch_mp_competitors_structures(csys, mpr, args.max_mp_competitors)

                comp_entries = entries_from_structures_with_chgnet(comp_structs)

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

                    decomp = pd.get_decomposition(ce)
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

                  "energy_above_hull", "is_stable", "decomposition"]

    with open(args.out, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for r in out_rows:

            writer.writerow(r)



    print(f"[OK] 写出 {len(out_rows)} 条记录 -> {args.out}")





if __name__ == "__main__":

    main()


