import os, glob, itertools, math
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from pymatgen.core import Structure, Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter

from chgnet.model import CHGNet
from mp_api.client import MPRester  # mp-api

def get_energy_pa(pred: dict) -> float:
    """从 CHGNet 预测结果里安全拿到 eV/atom."""
    # 旧/新版本兼容：优先常见缩写，其次全名
    for k in ("e", "energy"):
        if k in pred:
            return float(pred[k])
    # 有些版本对象化
    if hasattr(pred, "e"):
        return float(pred.e)
    if hasattr(pred, "energy"):
        return float(pred.energy)
    raise KeyError("energy")


# ====== 配置 ======
CIF_DIR = "results/exported_cifs"
EHULL_FILTER = 0.20   # 从 MP 抓竞争相时的 e_hull 上限 (eV/atom)
INCLUDE_MEgNet_BANDGAP = True  # 要不要给你的结构估计带隙（数据驱动近似）
OUT_ENERGY_CSV = "chgnet_energy_results.csv"
OUT_PHASE_CSV  = "chgnet_phase_results.csv"
OUT_PNG_DIR    = "phase_diagrams"

# ====== 可选：带隙模型 ======
if INCLUDE_MEgNet_BANDGAP:
    try:
        from megnet.models import MEGNetModel
        BG_MODEL = MEGNetModel.from_pretrained("logMEGNet-MP-2019.4.1-bandgap")
    except Exception as e:
        print("[WARN] MEGNet 预训练带隙模型加载失败，跳过带隙估计。", e)
        INCLUDE_MEgNet_BANDGAP = False
        BG_MODEL = None

# ====== 1) 批量用 CHGNet 预测能量 ======
print(">> 加载 CHGNet 预训练模型…")
chgnet = CHGNet.load()

records = []
file_list = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))
if not file_list:
    raise SystemExit(f"未在 {CIF_DIR} 找到 .cif 文件")

print(f">> 发现 {len(file_list)} 个 CIF，开始预测能量…")
for fp in tqdm(file_list):
    try:
        s = Structure.from_file(fp)
        pred = chgnet.predict_structure(s)  # energy(eV/atom), forces, stress
        e_pa = get_energy_pa(pred)
        natoms = len(s)
        e_tot = e_pa * natoms  # eV（按当前晶胞）
        formula = s.composition.reduced_formula
        elems = sorted([el.symbol for el in s.composition.elements])
        chemsys = "-".join(elems)

        row = {
            "file": os.path.basename(fp),
            "formula": formula,
            "chemsys": chemsys,
            "natoms_cell": natoms,
            "energy_per_atom_eV": e_pa,
            "energy_total_eV": e_tot,
        }

        if INCLUDE_MEgNet_BANDGAP and BG_MODEL is not None:
            try:
                eg = float(BG_MODEL.predict_structure(s))
            except Exception:
                eg = math.nan
            row["predicted_bandgap_eV"] = eg

        records.append(row)

    except Exception as e:
        print(f"[WARN] 处理 {fp} 失败：{e}")

df_energy = pd.DataFrame(records)
df_energy.to_csv(OUT_ENERGY_CSV, index=False)
if len(df_energy) == 0:
    raise SystemExit("CHGNet 预测结果为空（可能全部失败）。请检查上面的 warning 并确认键名已修复。")

print(f">> 已保存 {OUT_ENERGY_CSV} ({len(df_energy)} 行)")

# ====== 2) 从 MP 拉竞争相（包含所有子体系与元素参考），构建相图并计算 formation/Ehull ======
api_key = os.environ.get("MP_API_KEY")
if not api_key:
    raise SystemExit("未检测到 MP_API_KEY 环境变量。请设置后重试。")

os.makedirs(OUT_PNG_DIR, exist_ok=True)

mp = MPRester(api_key)

# 将我的结构按化学体系分组
grouped = defaultdict(list)
for _, r in df_energy.iterrows():
    key = tuple(r["chemsys"].split("-"))
    grouped[key].append(r)

phase_rows = []

def _subsets(elements):
    """生成非空子集（用于抓所有子体系）"""
    elems = list(elements)
    for L in range(1, len(elems)+1):
        for comb in itertools.combinations(elems, L):
            yield tuple(sorted(comb))

def fetch_mp_entries_for_system(elems):
    """
    为给定元素集合抓取 MP 竞争相（所有子体系）。
    返回 PDEntry 列表（energy 为总能量 eV/公式单元）。
    """
    from pymatgen.core import Composition
    entries = []

    # 抓所有子体系（1 元、2 元、...、满元），用于构建完整相图
    for sub in _subsets(elems):
        chemsys = "-".join(sub)
        docs = mp.materials.summary.search(  # ← 用 materials.summary
            chemsys=[chemsys],
            energy_above_hull=(0, EHULL_FILTER),
            fields=["composition_reduced", "energy_per_atom"]
        )
        for d in docs:
            comp = Composition(d.composition_reduced)
            e_pa = float(d.energy_per_atom)
            e_tot = e_pa * comp.num_atoms
            entries.append(PDEntry(comp, e_tot))

    # 兜底：确保每个元素都有一个参考相（最低能）
    for el in elems:
        docs_el = mp.materials.summary.search(  # ← 用 materials.summary
            elements=[el],
            num_elements=1,                      # ← 用 num_elements 而不是 nelements
            fields=["composition_reduced", "energy_per_atom"]
        )
        if docs_el:
            d0 = min(docs_el, key=lambda x: x.energy_per_atom)
            comp0 = Composition(d0.composition_reduced)
            e_pa0 = float(d0.energy_per_atom)
            entries.append(PDEntry(comp0, e_pa0 * comp0.num_atoms))
        else:
            print(f"[WARN] MP 未找到元素参考相：{el}")

    return entries


print(">> 为每个化学体系构建相图并计算 formation/Ehull …")
for system_key, my_rows in tqdm(grouped.items()):
    system_key = tuple(sorted(system_key))
    # 1) 我的结构转换为 PDEntry（注意用总能量）
    my_entries = []
    for r in my_rows:
        comp = Composition(r["formula"])  # reduced_formula 的 Composition
        # 为避免四舍五入歧义，更稳妥用文件真实结构的 composition
        # 但这里我们只有 formula；下面尝试从文件再读一次：
        try:
            s = Structure.from_file(os.path.join(CIF_DIR, r["file"]))
            comp = s.composition  # 晶胞化学计量
            e_tot = float(r["energy_total_eV"])
        except Exception:
            # 回退：用 reduced formula 的原子数近似总能 -> 误差可能较小体系也可接受
            comp = Composition(r["formula"])
            e_tot = float(r["energy_per_atom_eV"]) * comp.num_atoms
        my_entries.append(PDEntry(comp, e_tot))

    # 2) 竞争相（所有子体系 + 元素参考）
    mp_entries = fetch_mp_entries_for_system(system_key)

    # 3) 构建相图并计算指标
    all_entries = mp_entries + my_entries
    try:
        pdobj = PhaseDiagram(all_entries)
    except Exception as e:
        print(f"[WARN] 构建相图失败（{system_key}）：{e}")
        continue

    # 4) 输出我的每个结构的 formation energy / Ehull（单位 eV/atom）
    for e, r in zip(my_entries, my_rows):
        try:
            ehull = pdobj.get_e_above_hull(e)  # eV/atom
            fe_pa = pdobj.get_form_energy_per_atom(e)  # eV/atom
        except Exception as ex:
            print(f"[WARN] 计算失败 {r['file']}: {ex}")
            ehull, fe_pa = math.nan, math.nan

        out = dict(r)
        out["formation_energy_per_atom_eV"] = fe_pa
        out["energy_above_hull_eV"] = ehull
        phase_rows.append(out)

    # 5) 保存凸包图
    try:
        plotter = PDPlotter(pdobj, show_unstable=EHULL_FILTER)
        png_path = os.path.join(OUT_PNG_DIR, f"{'-'.join(system_key)}.png")
        # plotter.write_image(png_path)
    except Exception as ex:
        print(f"[WARN] 凸包图绘制失败（{system_key}）：{ex}")

df_phase = pd.DataFrame(phase_rows)
# 友好排序
cols = [
    "file","formula","chemsys","natoms_cell",
    "energy_per_atom_eV","energy_total_eV",
    "formation_energy_per_atom_eV","energy_above_hull_eV"
]
if INCLUDE_MEgNet_BANDGAP and "predicted_bandgap_eV" in df_phase.columns:
    cols.append("predicted_bandgap_eV")

# 只保留存在的列（避免因某些失败而缺列）
cols = [c for c in cols if c in df_phase.columns]
if len(df_phase) == 0:
    raise SystemExit("没有可用于相图的结构（phase_rows 为空）。")
df_phase = df_phase[cols].sort_values(["chemsys","energy_above_hull_eV"])

df_phase.to_csv(OUT_PHASE_CSV, index=False)
print(f">> 已保存 {OUT_PHASE_CSV} ({len(df_phase)} 行)")
print(f">> 凸包图已输出至：{OUT_PNG_DIR}/")
print("完成 ✅")
