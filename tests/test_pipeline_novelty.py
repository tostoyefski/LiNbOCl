"""The serial pipeline publishes final rows only after real dataset matching."""

import csv
import json
import sys
from pathlib import Path

from ase.io import write as ase_write
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "pipeline"))
sys.path.insert(0, str(ROOT / "mattergen"))

import run_top300_pipeline as pipeline


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rocksalt():
    return Structure(
        Lattice.cubic(5.6), ["Li"] * 4 + ["Cl"] * 4,
        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
         [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]],
    )


def test_serial_pipeline_applies_real_mattergen_novelty_after_voltage(tmp_path, monkeypatch):
    structures = {
        "known": rocksalt(),
        "unmatched": Structure(Lattice.cubic(3.4), ["Li", "Cl"],
                               [[0, 0, 0], [.5, .5, .5]]),
    }
    stage2 = tmp_path / "stage2.csv"
    source_rows = []
    for score, (name, structure) in zip((2, 1), structures.items()):
        source = tmp_path / f"{name}.extxyz"
        ase_write(source, AseAtomsAdaptor.get_atoms(structure), format="extxyz")
        source_rows.append({"path": str(source), "frame": 0, "quick_score": score,
                            "score_kind": "li_periodic_geometry_proxy_v1"})
    write_csv(stage2, source_rows)
    # Both local sources contain the known rocksalt structure; the other
    # polymorph has identical composition but distinct coordination.
    training, reference = tmp_path / "train.csv", tmp_path / "reference.csv"
    for path, identifier in ((training, "known-train"), (reference, "known-reference")):
        write_csv(path, [{"entry_id": identifier, "formula": "LiCl",
                          "structure": json.dumps(structures["known"].as_dict())}])
    output = tmp_path / "output"
    exported_rows = []
    events = []
    expected_relaxation = {"checkpoint": str(tmp_path / "custom-MatterSim.pth"),
                           "fmax": .02, "max_steps": 321}
    hull_settings = []
    snapshot = output / "relaxation" / "reference_entries.json"
    snapshot_contents = {"test_reference": "shared-with-voltage"}

    def export(export_script, refs_path, outdir, prefix, cwd):
        events.append("export")
        assert len(refs_path.read_text().splitlines()) == 2
        for name, structure in structures.items():
            cif = outdir / f"{name}.cif"
            structure.to(filename=str(cif))
            exported_rows.append({"ref": f"{tmp_path / (name + '.extxyz')}::0",
                                  "cif": str(cif)})
        index = outdir / "export_index.csv"
        write_csv(index, exported_rows)
        return index

    def hull(ehull_script, cif_dir, out_csv, cwd, index_csv=None,
             relaxation_settings=None):
        events.append("hull")
        assert index_csv.name == "export_300index.csv"
        assert len(read_rows(index_csv)) == 2
        assert relaxation_settings.as_dict() == expected_relaxation
        hull_settings.append(relaxation_settings)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(snapshot_contents))
        write_csv(out_csv, [{"file": Path(row["cif"]).name, "path": row["cif"],
                             "formula": "LiCl", "chemsys": "Cl-Li",
                             "calculation_status": "success", "hull_status": "complete",
                             "relaxation_status": "converged",
                             "energy_above_hull_eV": .01}
                            for row in exported_rows])

    def voltage(voltage_script, stable_csv, out_csv, voltage_step, threshold, cwd,
                target_voltage=None, relaxation_settings=None, reference_snapshot=None):
        events.append("voltage")
        assert len(read_rows(stable_csv)) == 2
        assert relaxation_settings is hull_settings[0]
        assert relaxation_settings.as_dict() == expected_relaxation
        assert reference_snapshot == snapshot
        assert json.loads(reference_snapshot.read_text()) == snapshot_contents
        write_csv(out_csv, [{"file": row["file"], "window_status": "stable_window",
                             "V_red": 1, "V_ox": 2, "window": 1}
                            for row in read_rows(stable_csv)])

    monkeypatch.setattr(pipeline, "run_export", export)
    monkeypatch.setattr(pipeline, "run_ehull", hull)
    monkeypatch.setattr(pipeline, "run_voltage", voltage)
    monkeypatch.chdir(tmp_path)

    pipeline.main([
        "--workdir", str(tmp_path), "--stage2-csv", str(stage2),
        "--output-dir", str(output), "--gpu-workers", "1",
        "--novelty-training-data", str(training),
        "--novelty-reference-data", str(reference),
        "--mattersim-checkpoint", expected_relaxation["checkpoint"],
        "--relax-fmax", str(expected_relaxation["fmax"]),
        "--relax-steps", str(expected_relaxation["max_steps"]),
    ])

    assert events == ["export", "hull", "voltage"]
    pre = read_rows(output / "pre_novelty_candidates.csv")
    assert [row["file"] for row in pre] == ["known.cif", "unmatched.cif"]
    assert all(row["passes_voltage_filter"] == "True" for row in pre)
    assert all("novelty_status" not in row for row in pre)
    final = read_rows(output / "final_candidates.csv")
    assert [row["file"] for row in final] == ["unmatched.cif"]
    assert final[0]["novelty_status"] == "unmatched"
    assert final[0]["passes_novelty_filter"] == "True"
    audit = {row["file"]: row for row in read_rows(output / "novelty_filter_audit.csv")}
    assert audit["known.cif"]["novelty_status"] == "matched"
    assert audit["known.cif"]["passes_novelty_filter"] == "False"
    assert json.loads(audit["known.cif"]["matched_reference_ids"]) == ["known-reference", "known-train"]
    summary = json.loads((output / "novelty_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["counts"]["input_candidates"] == 2
    assert summary["counts"]["matched"] == summary["counts"]["passed"] == 1
    assert summary["matcher"]["matcher_class"] == "mattergen.evaluation.utils.structure_matcher.DisorderedStructureMatcher"
    assert summary["matcher"]["matching_function"] == "mattergen.evaluation.utils.dataset_matcher.get_matches"
    assert summary["coverage_complete"] is True
    assert all(source["sha256"] for source in summary["sources"])
    assert training.is_file() and reference.is_file() and stage2.is_file()
