"""Validate coverage and numerical acceptance before writing the final package."""
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    report={};methods=['one-cut','recent','violated','angle','adaptive']
    for study,names,count in [('energy_verified',methods,75),('certificate',['fixed','certified','accurate'],80)]:
        checks=[]
        for seed in range(20):
            for name in names:
                p=HERE/'data'/study/f'run__{seed}__{name}.npz';z=np.load(p)
                assert len(z['iter'])==count and np.isfinite(z['energy_ref12']).all() and np.all(z['energy_ref12']>0)
                if study=='energy_verified':assert np.max(z['pgn'])<=1e-12
                if study=='certificate' and name=='certified':assert np.all(z['eps_hat']<=z['eta'])
                checks.append(str(p.name))
        report[study]=dict(completed=len(checks),expected=20*len(names),checks='PASS')
    total=0
    for n in [3000,6000,12000]:
        for seed in range(20):
            for name in methods:
                z=np.load(HERE/'data/scale'/f'{n}__{seed}__{name}.npz')
                assert len(z['normal'])==100 and np.isfinite(z['normal']).all();total+=1
    report['scale']=dict(completed=total,expected=300,checks='PASS')
    total=audits=0;zeros=0
    for problem in ['column_block','binary_digits']:
        for seed in range(20,40):
            for name in methods:
                root=HERE.parent/'extended_horizon_20260904/results' if seed<30 else HERE/'data/hard_new'
                path=root/f'{problem}__{seed}__{name}.npz';z=np.load(path)
                assert len(z['residual'])==10000 or z['residual'][-1]==0
                assert np.all(np.isfinite(z['residual'])) and np.all(z['residual']>=0);total+=1;zeros+=int(z['residual'][-1]==0)
                if problem=='binary_digits':
                    root=HERE.parent/'extended_horizon_20260904/mp_results' if seed<30 else HERE/'data/hard_mp'
                    rec=json.loads((root/f'binary_digits__{seed}__{name}__dps50.json').read_text())
                    assert rec['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                    r=float(z['residual'][-1]);m=float(rec['mp_saved_x_residual'])
                    assert abs(r-m)<=max(1e-12*abs(r),1e-100);audits+=1
    report['hard']=dict(completed=total,expected=200,zero_endpoints=zeros,checks='PASS')
    report['hard_mp50']=dict(completed=audits,expected=100,checks='PASS')
    mp=json.loads((HERE/'analysis/reference_mp60.json').read_text());assert mp['max_relative_energy_change']<1e-8 and mp['factor_change']<1e-10
    report['energy_mp60']=dict(instances=20,checks='PASS',**mp)
    for name in ['calibration','energy','gain','certificate','scale','hard']:
        assert (HERE/'deliverable/figures'/f'{name}.pdf').is_file()
    report['scope']='Experimental section only; full manuscript claims/references not re-audited.'
    (HERE/'analysis/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
