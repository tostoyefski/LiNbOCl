import json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from ase.io import read,write
from mattergen.scripts.generate import main as generate
from mattergen.evaluation.utils.relaxation import relax_structures
from pymatgen.io.ase import AseAtomsAdaptor
repo=Path('/home/litao/projects/LiNbOCl')
root=repo/'results/run6000_20261008'
shard=int(os.environ['SLURM_ARRAY_TASK_ID'])
out=root/'_segments'/f'batch{shard:03d}'/'Li-Nb-O-Cl'
out.mkdir(parents=True,exist_ok=True)
def status(stage,**extra):
    data=dict(stage=stage,shard=shard,updated=time.time(),conditions={'chemical_system':'Li-Nb-O-Cl','energy_above_hull':0.05},**extra)
    temp=out/'status.tmp'
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(out/'status.json')
    print(json.dumps(data),flush=True)
try:
    seed=2026100800+shard
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    status('generating',seed=seed,requested=200,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not (out/'generation_complete.json').exists():
        structures=generate(str(out),model_path='/home/litao/projects/mattergen-electrolytes-20261004/weights/chemical_system_energy_above_hull',batch_size=20,num_batches=10,
          properties_to_condition_on={'chemical_system':'Li-Nb-O-Cl','energy_above_hull':0.05},
          diffusion_guidance_factor=2.0,record_trajectories=False)
        if len(structures)!=200: raise RuntimeError(f'Expected 200 structures, got {len(structures)}')
        write(out/'generated_for_relaxation.extxyz',[AseAtomsAdaptor.get_atoms(s) for s in structures],format='extxyz')
        (out/'generation_complete.json').write_text(json.dumps({'count':200,'seed':seed}))
    atoms=read(out/'generated_for_relaxation.extxyz',index=':')
    if len(atoms)!=200: raise RuntimeError('Raw frame count mismatch')
    status('relaxing',generated=200)
    chunks=out/'relax_chunks'; chunks.mkdir(exist_ok=True)
    for start in range(0,len(atoms),20):
        dest=chunks/f'{start:04d}.extxyz'
        if not dest.exists():
            relaxed,energies=relax_structures([AseAtomsAdaptor.get_structure(a) for a in atoms[start:start+20]],device='cuda',potential_load_path=str(repo/'_runtime/MatterSim-v1.0.0-1M.pth'),output_path=str(dest)+'.tmp')
            if len(relaxed)!=20: raise RuntimeError('Relaxation frame count mismatch')
            Path(str(dest)+'.tmp').replace(dest)
        if len(read(dest,index=':'))!=20: raise RuntimeError(f'Invalid relaxation chunk {start}')
        status('relaxing',generated=200,relaxed=start+20)
    all_relaxed=[a for f in sorted(chunks.glob('*.extxyz')) for a in read(f,index=':')]
    if len(all_relaxed)!=200: raise RuntimeError('Final relaxed frame count mismatch')
    write(out/'relaxed.extxyz',all_relaxed,format='extxyz')
    status('completed',generated=200,relaxed=200)
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}')
    traceback.print_exc()
    raise
