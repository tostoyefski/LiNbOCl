"""Keep one relaxation protocol through CLI, Web, HPC and downstream geometry use."""
import csv
import importlib.util
from pathlib import Path
import shlex
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / 'workflow' / 'pipeline'
sys.path.insert(0, str(PIPELINE))
from mattersim_relaxation import RelaxationSettings
import run_top300_pipeline as pipeline


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def flag(command, name):
    return command[command.index(name) + 1]


def test_hull_and_voltage_receive_identical_nondefault_relaxation_settings(monkeypatch, tmp_path):
    settings = RelaxationSettings('/models/one shared model.pth', .03, 123)
    output = tmp_path / 'result.csv'
    commands = []
    def run(command, cwd=None):
        commands.append(command)
        output.touch()
    monkeypatch.setattr(pipeline, 'run_command', run)
    pipeline.run_ehull(PIPELINE / 'compute_ehull_chgnet.py', tmp_path, output, tmp_path,
                       relaxation_settings=settings)
    snapshot = tmp_path / 'relaxation/reference_entries.json'
    pipeline.run_voltage(PIPELINE / 'compute_voltage_window.py', tmp_path / 'stable.csv',
                         output, .05, .001, tmp_path, relaxation_settings=settings,
                         reference_snapshot=snapshot)
    for command in commands:
        assert flag(command, '--mattersim-checkpoint') == settings.checkpoint
        assert float(flag(command, '--relax-fmax')) == settings.fmax
        assert int(flag(command, '--relax-steps')) == settings.max_steps
    assert Path(flag(commands[1], '--reference-snapshot')) == snapshot


@pytest.mark.parametrize('arguments', [
    ['--relax-fmax', 'nan'], ['--relax-fmax', 'inf'], ['--relax-fmax', '0'],
    ['--relax-steps', '0'], ['--relax-steps', '-1'], ['--mattersim-checkpoint', ' '],
])
def test_top_cli_rejects_invalid_shared_protocol_before_starting_models(arguments):
    with pytest.raises(SystemExit) as error:
        pipeline.parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize('status_field,passed,failed', [
    ('calculation_status', 'success', 'calculation_failed'),
    ('hull_status', 'complete', 'calculation_failed'),
    ('relaxation_status', 'converged', 'failed'),
])
def test_failed_calculation_with_partial_finite_energy_cannot_pass_hull_gate(tmp_path, status_field, passed, failed):
    source, output = tmp_path / 'hull.csv', tmp_path / 'filtered.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'energy_above_hull_eV', status_field])
        writer.writeheader()
        writer.writerows([
            {'file': 'failed.cif', 'energy_above_hull_eV': 0, status_field: failed},
            {'file': 'converged.cif', 'energy_above_hull_eV': .02, status_field: passed},
        ])
    assert pipeline.filter_hull(source, output, .05) == 1
    with output.open() as handle:
        assert [row['file'] for row in csv.DictReader(handle)] == ['converged.cif']


def test_failed_rerun_removes_previous_final_pass_list(monkeypatch, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    final, audit = output / 'final_candidates.csv', output / 'voltage_filter_audit.csv'
    for path in (final, audit):
        path.write_text('previous passed candidate')
    monkeypatch.chdir(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('failed input')
    monkeypatch.setattr(pipeline, 'read_stage2_rows', fail)
    with pytest.raises(RuntimeError, match='failed input'):
        pipeline.main(['--workdir', str(tmp_path), '--stage2-csv', str(tmp_path / 'stage.csv'),
                       '--output-dir', str(output)])
    assert not final.exists() and not audit.exists()


def test_web_single_and_full_commands_use_same_protocol(monkeypatch, tmp_path):
    backend = load_module('uniform_entry_backend', REPO / 'mattergen_webapp/backend/main.py')
    calls = []
    monkeypatch.setattr(backend, 'launch_job', lambda kind, command, **kwargs: calls.append(command))
    request = backend.Top300Request(mattersim_checkpoint='/models/custom model.pth',
                                    relax_fmax=.03, relax_steps=123)
    backend.run_top300(request)
    single = calls[-1]
    backend.run_full(backend.FullPipelineRequest(
        dd=backend.GenerateRequest(base_results_dir=str(tmp_path / 'run')), top300=request))
    top_line = next(line for line in calls[-1][2].splitlines()
                    if '--stage2-csv' in line and 'run_top300_pipeline.py' in line)
    full = shlex.split(top_line)
    for command in (single, full):
        assert flag(command, '--mattersim-checkpoint') == '/models/custom model.pth'
        assert flag(command, '--relax-fmax') == '0.03'
        assert flag(command, '--relax-steps') == '123'


@pytest.mark.parametrize('changes', [{'relax_fmax': float('nan')}, {'relax_fmax': float('inf')},
                                     {'relax_fmax': 0}, {'relax_steps': 0}, {'mattersim_checkpoint': ' '}])
def test_web_rejects_invalid_relaxation_protocol(changes):
    backend = load_module('uniform_validation_backend', REPO / 'mattergen_webapp/backend/main.py')
    with pytest.raises(ValueError):
        backend.Top300Request(**changes)


def test_md_uses_audited_optimized_path_and_never_substitutes_original_export(tmp_path):
    md = load_module('uniform_md_paths', REPO / 'workflow/transport/compute_ionic_conductivity.py')
    original = tmp_path / 'exported_300cifs'
    original.mkdir()
    (original / 'candidate.cif').write_text('old geometry')
    optimized = tmp_path / 'relaxation/candidates/candidate.cif'
    row = {'file': 'candidate.cif', 'path': str(optimized)}
    assert md.candidate_cif_path(row, tmp_path / 'final.csv', original) == optimized
    assert not md.candidate_cif_path(row, tmp_path / 'final.csv', original).exists()
    assert md.candidate_cif_path({'file': 'candidate.cif'}, tmp_path / 'final.csv', original) == original / 'candidate.cif'
    relative = {'file': 'candidate.cif', 'path': 'relaxation/candidates/candidate.cif'}
    assert md.candidate_cif_path(relative, tmp_path / 'final.csv', original) == optimized
    with pytest.raises(ValueError, match='does not match'):
        md.candidate_cif_path({'file': 'candidate.cif', 'path': str(original / 'other.cif')},
                              tmp_path / 'final.csv', original)
