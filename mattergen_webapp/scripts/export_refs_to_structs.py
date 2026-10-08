#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

从 refs 列表 (/abs/path/relaxed.extxyz::frame) 批量导出结构为独立文件。

默认导出 CIF，可选 --poscar 一并导出 POSCAR。

生成 export_index.csv 记录 ref -> 输出文件 的映射与元信息。

"""

import os, re, argparse, csv

from pathlib import Path

from tqdm import tqdm

import pandas as pd

from ase.io import read as ase_read, write as ase_write

from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



def parse_ref(line: str):

    """解析一行 /path/to/relaxed.extxyz::frame_index"""

    line = line.strip()

    if not line or line.startswith("#"):

        return None, None

    if "::" not in line:

        raise ValueError(f"非法引用（缺少 '::'）: {line}")

    path, idx = line.split("::", 1)

    path = path.strip()

    idx = idx.strip()

    if not idx.isdigit():

        raise ValueError(f"非法帧编号: {line}")

    return path, int(idx)



def safe_name(s: str) -> str:

    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)



def read_frame(path: str, idx: int):

    """读取 extxyz 指定帧（0-based）"""

    return ase_read(path, index=idx)



def summarize(atoms):

    """返回 formula, spg, n_atoms, density"""

    struct = AseAtomsAdaptor().get_structure(atoms)

    formula = struct.composition.reduced_formula

    try:

        spg = SpacegroupAnalyzer(struct, symprec=0.2, angle_tolerance=5).get_space_group_symbol()

    except Exception:

        spg = "P1"

    return {

        "formula": formula,

        "spg": spg,

        "n_atoms": struct.num_sites,

        "density": round(struct.density, 5),

        "structure": struct,

    }



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--refs", type=str, default="top150_refs.txt",

                    help="包含 path::frame 的列表（每行一个）")

    ap.add_argument("--outdir", type=str, default="exported_cifs",

                    help="导出目录（不存在将创建）")

    ap.add_argument("--prefix", type=str, default="cand",

                    help="导出文件名前缀")

    ap.add_argument("--poscar", action="store_true",

                    help="同时导出 POSCAR（默认只导出 CIF）")

    args = ap.parse_args()



    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)



    refs_path = Path(args.refs)

    if not refs_path.exists():

        raise FileNotFoundError(f"未找到引用列表: {refs_path}")



    rows = []

    total = 0

    with refs_path.open() as f:

        lines = f.readlines()



    for line in tqdm(lines, desc="Exporting"):

        path, frame = parse_ref(line)

        if path is None:

            continue

        if not Path(path).exists():

            print(f"[WARN] 跳过，不存在: {path}")

            continue

        try:

            atoms = read_frame(path, frame)

        except Exception as e:

            print(f"[WARN] 读取失败: {path}::{frame} ({e})")

            continue



        meta = summarize(atoms)

        # 生成唯一文件名：prefix_idx_formula_spg

        base = f"{args.prefix}_{total:04d}_{meta['formula']}_{meta['spg']}"

        base = safe_name(base)

        cif_path = outdir / f"{base}.cif"

        # 导出 CIF（用 pymatgen 写更标准；此处直接用 ASE 也可）

        try:

            # 用 pymatgen 写 CIF

            from pymatgen.io.cif import CifWriter

            CifWriter(meta["structure"], symprec=None).write_file(str(cif_path))

        except Exception:

            # 退回 ASE 写

            ase_write(str(cif_path), atoms, format="cif")



        poscar_path = ""

        if args.poscar:

            poscar_path = outdir / f"{base}.vasp"

            try:

                ase_write(str(poscar_path), atoms, format="vasp")

            except Exception as e:

                print(f"[WARN] 写 POSCAR 失败: {poscar_path} ({e})")

                poscar_path = ""



        rows.append({

            "ref": f"{path}::{frame}",

            "cif": str(cif_path),

            "poscar": str(poscar_path) if poscar_path else "",

            "formula": meta["formula"],

            "spg": meta["spg"],

            "n_atoms": meta["n_atoms"],

            "density": meta["density"],

        })

        total += 1



    if not rows:

        print("没有成功导出的结构。")

        return



    # 写索引表

    idx_csv = outdir / "export_index.csv"

    pd.DataFrame(rows).to_csv(idx_csv, index=False)

    print(f"\n完成：导出 {total} 个结构到 {outdir}/")

    print(f"索引表：{idx_csv}")



if __name__ == "__main__":

    main()

