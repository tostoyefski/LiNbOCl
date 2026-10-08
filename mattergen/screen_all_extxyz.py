#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

批量快筛多帧 relaxed.extxyz：

- 递归遍历 base 目录

- 对每个 relaxed.extxyz 读取**所有帧**（每帧视作一个不同材料）

- 计算：密度、组成特征（Li 分数、O/(O+卤)）、卤素混合熵、Li–Li 连通性（2x2x2 超胞）

- 可选：价态/电荷平衡（pymatgen 方式）与 SMACT 规则筛

- 生成 stage2_candidates.csv（按 quick_score 排序），并写出 top150_refs.txt（格式：path::frame）

"""

import os, argparse, math, warnings

from glob import iglob

from tqdm import tqdm

import numpy as np

import pandas as pd



from ase.io import iread as ase_iread

from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.core import Composition

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:

    from smact.screening import pauling_test as smact_pauling_test

except Exception:

    smact_pauling_test = None



HALS = ("F","Cl","Br","I")



def iter_frames(path):

    """惰性读取 extxyz 帧，避免一次性载入内存。"""

    yield from ase_iread(path, index=":")



def structure_from_atoms(atoms):

    return AseAtomsAdaptor().get_structure(atoms)



def comp_features(struct):

    comp = struct.composition

    el = comp.get_el_amt_dict()

    n_li = el.get("Li", 0.0)

    n_o  = el.get("O", 0.0)

    n_hal= sum(el.get(x,0.0) for x in HALS)

    n_tot= sum(el.values()) if el else 0.0

    f_li = (n_li/n_tot) if n_tot>0 else 0.0

    f_o  = (n_o/(n_o+n_hal)) if (n_o+n_hal)>0 else 0.0

    # 卤素混合熵（香农熵，单位: nat）

    hal_counts = [el.get(x,0.0) for x in HALS]

    hal_tot = sum(hal_counts)

    if hal_tot>0:

        p = [c/hal_tot for c in hal_counts if c>0]

        hal_entropy = -sum(pi*math.log(pi) for pi in p)

    else:

        hal_entropy = 0.0

    return {

        "formula": comp.reduced_formula,

        "n_atoms": struct.num_sites,

        "density": struct.density,

        "f_li": f_li,

        "f_o": f_o,

        "hal_entropy": hal_entropy,

    }



def li_connectivity(struct, r_cut=3.0, super=(2,2,2)):

    """简易 Li–Li 连通代理：返回 li_conn 与最小 Li–Li 距离。"""

    sup = struct * super

    li_idx = [i for i,site in enumerate(sup) if site.specie.symbol=="Li"]

    if len(li_idx) < 2:

        return {"li_conn": 0.0, "min_li_li": 999.0}



    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        dmat = sup.distance_matrix

    sub = dmat[np.ix_(li_idx, li_idx)].copy()

    np.fill_diagonal(sub, 1e9)



    deg = (sub < r_cut).sum(axis=1)

    li_conn    = float((deg >= 2).mean())

    min_li_li  = float(sub.min()) if sub.size>0 else 999.0

    return {"li_conn": li_conn, "min_li_li": min_li_li}



def soft_clip(x, lo, hi):

    if x is None: return 0.0

    return max(0.0, min(1.0, (x - lo) / (hi - lo + 1e-9)))



def quick_score_row(row):

    # Li 通道

    s_li  = 0.6*soft_clip(row["li_conn"], 0.30, 0.90) + 0.4*soft_clip(row["min_li_li"], 1.9, 3.0)

    # 轻度氧化/混卤/密度

    s_o   = 1.0 - abs(row["f_o"] - 0.20)/0.20

    s_mix = 1.0 - abs(row["hal_entropy"] - 0.70)/0.70

    s_rho = 1.0 - abs(row["density"] - 2.60)/1.00

    s_o   = max(0.0, min(1.0, s_o))

    s_mix = max(0.0, min(1.0, s_mix))

    s_rho = max(0.0, min(1.0, s_rho))

    return 0.35*s_li + 0.25*(0.5*s_o + 0.3*s_mix + 0.2*s_rho)



# ---- 可选：价态/电荷平衡 & SMACT 快筛 ----

def charge_balance_ok(comp: Composition) -> bool:

    try:

        guesses = comp.oxi_state_guesses(all_oxi_states=True)

        return bool(guesses)

    except Exception:

        return True  # 防止误杀



def smact_ok(comp: Composition) -> bool:

    if smact_pauling_test is None:

        return True

    try:

        # pauling_test 接受元素符号列表；它较保守，可能误杀少量边界案例

        syms = [el.symbol for el in comp.elements]

        return bool(smact_pauling_test(syms))

    except Exception:

        return True



def guess_spg(struct):

    try:

        return SpacegroupAnalyzer(struct, symprec=0.2, angle_tolerance=5).get_space_group_symbol()

    except Exception:

        return "P1"



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--base", type=str, default="results/chemical_system_energy_above_hull",

                    help="递归查找 relaxed.extxyz 的根目录")

    ap.add_argument("--out", type=str, default="stage2_candidates.csv", help="输出 CSV 路径")

    ap.add_argument("--r-cut", type=float, default=3.0, help="Li–Li 连通性的截断 (Å)")

    ap.add_argument("--super", type=int, nargs=3, default=(2,2,2), help="超胞复制，例如 2 2 2")

    ap.add_argument("--light-oxy", type=float, nargs=2, default=(0.05, 0.35),

                    help="轻度氧化判定阈值区间 for f_o")

    ap.add_argument("--require-charge-balance", action="store_true",

                    help="启用：仅保留可电中性的配方（pymatgen 估计）")

    ap.add_argument("--use-smact", action="store_true",

                    help="启用：结合 SMACT 的保守化学规则筛（需 pip install smact）")

    args = ap.parse_args()



    rows = []

    screened_out = []

    cb_cache = {}

    sm_cache = {}

    found_any = False

    for p in tqdm(iglob(os.path.join(args.base, "**", "relaxed.extxyz"), recursive=True), desc="Scanning files"):

        found_any = True

        try:

            for idx, atoms in enumerate(iter_frames(p)):

                try:

                    s = structure_from_atoms(atoms)

                    feats = comp_features(s)

                    formula_key = feats["formula"]

                    if formula_key not in cb_cache:

                        cb_cache[formula_key] = charge_balance_ok(s.composition)

                    cb_ok = cb_cache[formula_key]

                    if formula_key not in sm_cache:

                        sm_cache[formula_key] = smact_ok(s.composition)  # 若未安装 smact，函数内会返回 True

                    sm_ok = sm_cache[formula_key]



                    # 根据开关决定是否过滤，并记录原因

                    filtered = False

                    reasons = []

                    if args.require_charge_balance and not cb_ok:

                        filtered = True

                        reasons.append("no_charge_balance")

                    if args.use_smact and not sm_ok:

                        filtered = True

                        reasons.append("smact_fail")



                    conn  = li_connectivity(s, r_cut=args.r_cut, super=tuple(args.super))

                    spg   = guess_spg(s)



                    base_row = {

                        "path": p,

                        "frame": idx,

                        "dir": os.path.dirname(p),

                        "spg": spg,

                        **feats,

                        **conn,

                        "passes_light_oxy": (args.light_oxy[0] <= feats["f_o"] <= args.light_oxy[1]),

                        "charge_balance_ok": cb_ok,

                        "smact_ok": sm_ok,

                    }

                    base_row["quick_score"] = quick_score_row(base_row)



                    # 若被过滤，写入 screened_out；否则进入正式结果

                    if filtered:

                        base_row["filtered_reasons"] = ",".join(reasons)

                        screened_out.append(base_row)

                        continue



                    rows.append(base_row)



                except Exception as e:

                    print(f"[SKIP] 解析失败: {p}#frame{idx} ({e})")



        except Exception as e:

            print(f"[SKIP] 读取失败: {p} ({e})")

            continue





    if not found_any:

        print(f"[WARN] 没在 {args.base} 下找到任何 relaxed.extxyz")

        return

    if not rows:

        print("[ERROR] 没有成功解析的帧。")

        return



    df = pd.DataFrame(rows)

    df.sort_values("quick_score", ascending=False, inplace=True)

    df.to_csv(args.out, index=False)

    # 导出被过滤样品（若有）

    if screened_out:

        pd.DataFrame(screened_out).to_csv("screened_out.csv", index=False)

        print(f"已导出被过滤样品: screened_out.csv  （含 filtered_reasons 列）")





    # 也输出 top-150 的“引用列表”，包含 path::frame，便于后续精筛或导出

    topk = min(150, len(df))

    refs = (df.head(topk)["path"] + "::" + df.head(topk)["frame"].astype(str)).tolist()

    with open("top150_refs.txt", "w") as f:

        f.write("\n".join(refs))



    print(f"\n完成: 共 {len(df)} 帧 -> {args.out}")

    print("Top-10 预览：")

    print(df[["formula","spg","density","f_li","f_o","hal_entropy","li_conn","min_li_li","quick_score","path","frame"]].head(10))

    print("Top-150 引用列表写入: top150_refs.txt  （格式：/path/to/relaxed.extxyz::frame_index）")



if __name__ == "__main__":

    main()
