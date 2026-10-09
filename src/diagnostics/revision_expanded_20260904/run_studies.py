import argparse
import json
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(HERE/'engine'))
import experiments_adaptive_revision as ad
import experiments_bounded as base
import numerics
import numpy as np
from threadpoolctl import threadpool_limits

def save(stem,h,metadata):
    np.savez_compressed(stem.with_suffix('.npz'),**h)
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',choices=['energy','certificate','scale']);ap.add_argument('--seeds',nargs='+',type=int,default=list(range(20)))
    ap.add_argument('--verify-inner',action='store_true')
    args=ap.parse_args();out=HERE/'data'/('energy_verified' if args.study=='energy' and args.verify_inner else args.study);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((HERE/'experiments/results_adaptive_revision/selected_configuration.json').read_text())
    ad.bregman=base.bregman=numerics.bregman
    inner_audit=[]
    if args.verify_inner:
        if args.study!='energy':raise ValueError('Verified polishing is scoped to the energy study')
        original_solver=ad._dual_solver
        def verified_solver(s,G,beta,tau,**kwargs):
            alpha,info=original_solver(s,G,beta,tau,**kwargs)
            tol=kwargs['tol'];before=info['pgn'];extra=0
            if before>tol:
                step=1./max(float(np.linalg.norm(G,2)**2),1e-16)
                for extra in range(1,20001):
                    _,g=ad.dual_phi_grad(alpha,s,G,beta,tau)
                    alpha=np.maximum(alpha-step*g,0.)
                    _,g=ad.dual_phi_grad(alpha,s,G,beta,tau)
                    pgn=ad.projected_grad_norm(alpha,g)
                    if pgn<=tol:break
                if pgn>tol:raise FloatingPointError(f'Energy inner accuracy unmet: {pgn} > {tol}')
                info=dict(info,pgn=pgn,nit=info['nit']+extra,success=True)
            inner_audit.append(dict(before=before,after=info['pgn'],polish_iterations=extra))
            return alpha,info
        ad._dual_solver=verified_solver
    methods=ad._comparison_methods(selection)
    deadline=time.perf_counter()+1800
    with threadpool_limits(1):
        numerics.test()
        for seed in args.seeds:
            if time.perf_counter()>deadline:raise TimeoutError('30-minute study batch limit')
            if args.study=='scale':
                for m,n in [(1000,3000),(2000,6000),(4000,12000)]:
                    A,b,tau=base.make_sparse_problem(10100+n+seed,m,n);L=ad.spectral_lipschitz(A)
                    for method in methods:
                        stem=out/f'{n}__{seed}__{method.name}'
                        if stem.with_suffix('.json').exists():continue
                        print(f'SCALE {n} {seed} {method.name}',flush=True)
                        h=ad._timed_run(A,b,tau,method,100,L=L,repeats=3)
                        save(stem,h,dict(seed=seed,m=m,n=n,method=method.name,selection=selection))
                continue
            source_seed=(9100 if args.study=='energy' else 5000)+seed
            A,b,_,tau=base.make_gaussian_problem(source_seed,60,180,sparsity=12);L=ad.spectral_lipschitz(A)
            refpath=out/f'reference__{seed}.npz'
            if refpath.exists():
                z=np.load(refpath);refs=[z['ref10'],z['ref12']];diags=json.loads(refpath.with_suffix('.json').read_text())
            else:
                print(f'REFERENCE {args.study} {seed}',flush=True)
                pairs=[numerics.reference(A,b,tau,t) for t in [1e-10,1e-12]]
                refs=[r[0] for r in pairs];diags=[r[1] for r in pairs]
                np.savez_compressed(refpath,ref10=refs[0],ref12=refs[1],A=A,b=b,tau=tau)
                refpath.with_suffix('.json').write_text(json.dumps(diags,indent=2),encoding='utf-8')
            configurations=methods if args.study=='energy' else ['fixed','certified','accurate']
            for item in configurations:
                name=item.name if args.study=='energy' else item
                stem=out/f'run__{seed}__{name}'
                if stem.with_suffix('.json').exists():continue
                print(f'{args.study.upper()} {seed} {name}',flush=True)
                inner_audit.clear()
                if args.study=='energy':
                    # Solver tolerance is identical across every memory policy.
                    h=ad.run_method(A,b,tau,item,75,L=L,xbar=refs[1],track_theta=True,dual_tol=1e-12)
                else:
                    method=base.Method('violated','violated',selection['qmax'],selection['pmax'])
                    h=base.run_algorithm(A,b,tau,method,80,L=L,xbar=refs[1],solve_mode=item,fixed_tol=1e-3,pool_size=selection['pmax'])
                    if item!='certified':h['eta']=1e-3*np.linalg.norm(A.T@b)**2/(2*L*L)/(np.arange(len(h['iter']))+1)**1.5
                E0=[numerics.bregman(r,np.zeros(180),np.zeros(180),tau) for r in refs]
                for j in [0,1]:
                    h[f'energy_ref{10 if j==0 else 12}']=np.array([numerics.bregman(refs[j],x,s,tau)/E0[j] for x,s in zip(h['primal_trace'],h['dual_trace'])])
                h['energy_direct_raw']=np.array([numerics.direct_energy(refs[1],x,s,tau) for x,s in zip(h['primal_trace'],h['dual_trace'])])
                metadata=dict(seed=seed,source_seed=source_seed,method=name,selection=selection,
                              verified_inner=args.verify_inner,inner_audit=inner_audit.copy(),
                              reference_diags=diags,minimum_raw_direct_energy=float(h['energy_direct_raw'].min()),
                              minimum_stable_energy=float(h['energy_ref12'].min()*E0[1]),
                              max_reference_relative_energy_change=float(np.max(np.abs(h['energy_ref10']-h['energy_ref12'])/np.maximum(h['energy_ref12'],1e-300))))
                save(stem,h,metadata)
    print('COMPLETED',args.study,args.seeds,flush=True)

if __name__=='__main__':main()
