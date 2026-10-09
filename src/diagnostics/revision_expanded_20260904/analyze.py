import argparse
import csv
import json
from pathlib import Path
import sys
import os
HERE=Path(__file__).resolve().parent
os.environ['MPLBACKEND']='Agg';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']=str(HERE/'.matplotlib')
sys.path.insert(0,str(HERE/'engine'))
import numpy as np
import matplotlib.pyplot as plt
import experiments_adaptive_revision as ad
from threadpoolctl import threadpool_limits
OUT=HERE/'analysis';FIG=HERE/'deliverable/figures'
NAMES=['one-cut','recent','violated','angle','adaptive']
STYLES=['-','--','-.',':','-']

def csvwrite(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(v):
    v=np.asarray(v,dtype=float)
    return dict(mean=float(np.mean(v)),median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)))

def readcsv(name):
    return list(csv.DictReader((HERE/'experiments/results_adaptive_revision'/name).open()))

def savefig(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def band(ax,k,v,name,color,style='-',floor=1e-30):
    ax.plot(k,np.maximum(np.median(v,axis=0),floor),style,color=color,label=name,lw=1.6)
    ax.fill_between(k,np.maximum(np.quantile(v,.25,axis=0),floor),np.maximum(np.quantile(v,.75,axis=0),floor),color=color,alpha=.11,lw=0)
    ax.set_yscale('log');ax.grid(alpha=.2);ax.set_xlabel('Iteration')

def fit(e):
    k=np.arange(40,76);e=np.asarray(e)[39:75]
    if len(e)!=36 or np.any(e<=0):raise ValueError('Invalid fixed fit window')
    y=np.log(e);slope,intercept=np.polyfit(k,y,1);rss=float(np.sum((y-(slope*k+intercept))**2));tss=float(np.sum((y-y.mean())**2))
    return float(np.exp(slope)),1-rss/tss if tss else np.nan

def core():
    result={}
    selection=json.loads((HERE/'experiments/results_adaptive_revision/selected_configuration.json').read_text());result['selection']=selection
    fig,axes=plt.subplots(2,2,figsize=(8.4,5.1))
    sr=readcsv('exp1_solver_calibration.csv');qr=readcsv('exp1_q_calibration.csv');pr=readcsv('exp1_p_calibration.csv');cr=readcsv('exp1_trigger_calibration.csv')
    result['solver_calibration']=sr;result['q_calibration']=qr;result['p_calibration']=pr;result['trigger_calibration']=cr
    axes[0,0].bar(['L-BFGS-B','PG','Accel. PG','Proj. BB'],[float(r['runtime']) for r in sr],color='#7AA6C2');axes[0,0].set_ylabel('Mean time (s)');axes[0,0].set_title('Inner solver')
    for ax,rows,key,title in [(axes[0,1],qr,'qmax','Working set'),(axes[1,0],pr,'pmax','Candidate pool'),(axes[1,1],cr,'c_trig','Trigger')]:
        x=[float(r[key]) for r in rows];v=[float(r['final_residual']) for r in rows]
        ax.plot(x,v,'o-',color='#0072B2');ax.set_yscale('log');ax.set_xlabel({'qmax':'q','pmax':'p','c_trig':'Trigger threshold'}[key]);ax.set_ylabel('Mean residual');ax.set_title(title)
        selected=float(selection[key]);idx=x.index(selected);ax.scatter([x[idx]],[v[idx]],s=100,facecolor='none',edgecolor='black')
        if key=='c_trig':ax.set_xscale('symlog',linthresh=.001)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    savefig(fig,'calibration')
    p=HERE/'data/energy_verified';runs={};rr=[];refs=[]
    for seed in range(20):
        for d in json.loads((p/f'reference__{seed}.json').read_text()):refs.append(dict(study='energy',seed=seed,**d))
        for name in NAMES:
            z=np.load(p/f'run__{seed}__{name}.npz');runs[seed,name]={k:z[k] for k in z.files}
            q,r2=fit(z['energy_ref12']);q10,r210=fit(z['energy_ref10'])
            metadata=json.loads((p/f'run__{seed}__{name}.json').read_text())
            refdata=np.load(p/f'reference__{seed}.npz');ref=refdata['ref12'];tau=float(refdata['tau'])
            e0=tau*np.linalg.norm(ref,1)+.5*(ref@ref)
            rr.append(dict(seed=seed,method=name,final_energy=float(z['energy_ref12'][-1]),factor=q,r_squared=r2,factor_ref10=q10,
                           max_reference_relative_change=metadata['max_reference_relative_energy_change'],min_raw_direct=metadata['minimum_raw_direct_energy'],
                           max_direct_vs_stable_relative=float(np.max(np.abs(z['energy_direct_raw']/e0-z['energy_ref12'])/np.maximum(z['energy_ref12'],1e-300))),
                           actual_max_pgn=float(z['pgn'].max()),requested_tol=1e-12))
    csvwrite('energy_per_instance.csv',rr)
    result['energy']=[dict(method=name,final=stats([r['final_energy'] for r in rr if r['method']==name]),factor=stats([r['factor'] for r in rr if r['method']==name]),r_squared=stats([r['r_squared'] for r in rr if r['method']==name])) for name in NAMES]
    result['energy_accuracy']=dict(max_reference_change=max(r['max_reference_relative_change'] for r in rr),max_factor_change=max(abs(r['factor']-r['factor_ref10']) for r in rr),min_raw_direct=min(r['min_raw_direct'] for r in rr),min_displayed_energy=min(float(runs[s,m]['energy_ref12'][39:75].min()) for s in range(20) for m in NAMES))
    fig,ax=plt.subplots(figsize=(6.5,3.3))
    for name,style in zip(NAMES,STYLES):band(ax,np.arange(40,76),np.stack([runs[s,name]['energy_ref12'][39:75] for s in range(20)]),name,ad.COLORS[name],style)
    ax.set_ylabel('Relative Bregman energy');ax.legend(ncol=3,fontsize=8);ax.set_xlim(40,75);savefig(fig,'energy')
    theta=np.stack([runs[s,'violated']['theta'][:55] for s in range(20)]);lower=np.stack([runs[s,'violated']['theta_lb'][:55] for s in range(20)])
    valid=np.isfinite(theta)&np.isfinite(lower);margin=(theta-lower)[valid]
    result['gain']=dict(min_margin=float(margin.min()),violations=int(np.sum(margin < -1e-9)),evaluated=int(valid.sum()),max_mean_theta=float(np.nanmean(theta,axis=0).max()))
    csvwrite('gain_per_iteration.csv',[dict(seed=s,iteration=k+1,theta=float(theta[s,k]),lower=float(lower[s,k])) for s in range(20) for k in range(55)])
    fig,ax=plt.subplots(figsize=(6.2,3.1));k=np.arange(1,56)
    ax.plot(k,np.nanmean(theta,axis=0),label='Observed gain',color='#0072B2');ax.plot(k,np.nanmean(lower,axis=0),'--',label='Lower bound',color='#D55E00');ax.axhline(1,color='black',lw=.8,label='One-cut');ax.set_xlabel('Iteration');ax.set_ylabel('Gain factor');ax.legend(fontsize=8);ax.grid(alpha=.2);savefig(fig,'gain')
    p=HERE/'data/certificate';rr=[];cruns={}
    for seed in range(20):
        for d in json.loads((p/f'reference__{seed}.json').read_text()):refs.append(dict(study='certificate',seed=seed,**d))
        for name in ['fixed','certified','accurate']:
            z=np.load(p/f'run__{seed}__{name}.npz');cruns[seed,name]={k:z[k] for k in z.files}
            violations=int(np.sum(z['eps_hat']>z['eta']*(1+1e-8)))
            if name=='certified':assert violations==0
            rr.append(dict(seed=seed,rule=name,final_energy=float(z['energy_ref12'][-1]),runtime=float(z['runtime'][-1]),mean_inner=float(z['dual_nit'].mean()),budget_violations=violations,max_certificate_ratio=float(np.max(z['eps_hat']/z['eta']))))
    csvwrite('certificate_per_instance.csv',rr);csvwrite('reference_accuracy.csv',refs)
    result['references']=dict(max_feasibility=max(r['primal_feasibility'] for r in refs),max_normal_residual=max(r['normal_residual'] for r in refs),max_gap=max(r['primal_dual_gap'] for r in refs),max_reference_distance_bound=max(r['reference_distance_bound'] for r in refs),refine_success=all(r['refine_success'] for r in refs))
    result['certificate']=[dict(rule=name,**{key:stats([r[key] for r in rr if r['rule']==name]) for key in ['final_energy','runtime','mean_inner','budget_violations']}) for name in ['fixed','certified','accurate']]
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2))
    for name,label,color in [('fixed','Fixed PG','#D55E00'),('certified','Certified','#009E73'),('accurate','High accuracy','#0072B2')]:
        energy=np.stack([cruns[s,name]['energy_ref12'] for s in range(20)]);ratios=np.stack([cruns[s,name]['eps_hat']/cruns[s,name]['eta'] for s in range(20)])
        band(axes[0],np.arange(1,81),energy,label,color);band(axes[1],np.arange(1,81),ratios,label,color,floor=1e-18)
    axes[0].set_ylabel('Relative Bregman energy');axes[1].set_ylabel(r'$\widehat{\varepsilon}_k/\eta_k$');axes[1].axhline(1,color='black',ls=':',lw=1);axes[0].legend(fontsize=8)
    savefig(fig,'certificate');return result

def scale():
    p=HERE/'data/scale';rr=[];fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,n in zip(axes,[3000,6000,12000]):
        for name,style in zip(NAMES,STYLES):
            vals=[]
            for seed in range(20):
                z=np.load(p/f'{n}__{seed}__{name}.npz');vals.append(z['normal']);hit=ad._persistent_time(z,5e-4)
                rr.append(dict(n=n,seed=seed,method=name,final_residual=float(z['normal'][-1]),runtime=float(z['runtime'][-1]),target_time=float(hit),success=int(np.isfinite(hit)),q1_share=float(np.mean(z['memory_size']==1)),mean_q=float(np.mean(z['memory_size']))))
            band(ax,np.arange(1,101),np.stack(vals),name,ad.COLORS[name],style)
        ax.axhline(5e-4,color='gray',ls=':',lw=1);ax.set_title(f'n = {n:,}');ax.set_ylabel('Relative residual')
    axes[0].legend(fontsize=7);savefig(fig,'scale');csvwrite('scale_per_instance.csv',rr)
    return [dict(n=n,method=name,final=stats([r['final_residual'] for r in rr if r['n']==n and r['method']==name]),runtime=stats([r['runtime'] for r in rr if r['n']==n and r['method']==name]),success=sum(r['success'] for r in rr if r['n']==n and r['method']==name),q1_share=stats([r['q1_share'] for r in rr if r['n']==n and r['method']==name])) for n in [3000,6000,12000] for name in NAMES]

def hard():
    previous=HERE.parent/'extended_horizon_20260904/results';new=HERE/'data/hard_new';rr=[];curves={}
    for p in ['column_block','binary_digits']:
        for seed in range(20,40):
            for name in NAMES:
                src=previous if seed<30 else new;z=np.load(src/f'{p}__{seed}__{name}.npz');v=z['residual'];zero=bool(v[-1]==0)
                assert zero or len(v)==10000
                full=np.pad(v,(0,10000-len(v)),constant_values=0) if zero else v
                curves[p,seed,name]=full
                a=float(np.median(full[4000:5000]));b=float(np.median(full[9000:10000]));drift=abs(a-b)/max(a,b,np.finfo(float).tiny)
                rr.append(dict(problem=p,seed=seed,method=name,cohort='previous' if seed<30 else 'new',terminal=float(b),final=float(v[-1]),iterations=len(v),zero=int(zero),checkpoint_drift=drift,stable=int(drift<.01),unchanged=int(zero or np.all(z['update_norm'][-1000:]==0)),tail_inner_pass=float(np.mean(z['inner_pass'][-1000:]))))
    csvwrite('hard_per_instance.csv',rr);summary=[]
    for cohort in ['all','previous','new']:
        for p in ['column_block','binary_digits']:
            subset=[r for r in rr if r['problem']==p and (cohort=='all' or r['cohort']==cohort)]
            baseline={r['seed']:r['terminal'] for r in subset if r['method']=='one-cut'}
            for name in NAMES:
                rows=[r for r in subset if r['method']==name]
                summary.append(dict(problem=p,method=name,cohort=cohort,n=len(rows),**stats([r['terminal'] for r in rows]),lower=sum(r['terminal']<baseline[r['seed']] for r in rows),zero=sum(r['zero'] for r in rows),stable=sum(r['stable'] for r in rows),unchanged=sum(r['unchanged'] for r in rows)))
    csvwrite('hard_summary.csv',summary)
    minimum=min(float(v[v>0].min()) for v in curves.values() if np.any(v>0));floor=10**(np.floor(np.log10(minimum))-1)
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.5));idx=np.arange(10000)
    for ax,p,title in zip(axes,['column_block','binary_digits'],['Column-block completion','Binary Digits']):
        for name,style in zip(NAMES,STYLES):
            band(ax,idx+1,np.stack([curves[p,s,name][idx] for s in range(20,40)]),name,ad.COLORS[name],style,floor=floor)
        ax.set_xscale('log');ax.set_xlabel('Iteration (log scale)');ax.set_ylabel('Relative residual');ax.set_title(title)
    axes[0].legend(ncol=2,fontsize=7);savefig(fig,'hard');return dict(rows=summary,zero_display_floor=floor)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--core-only',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    with threadpool_limits(1):
        result=core()
        if not args.core_only:result['scale']=scale();result['hard']=hard()
    (OUT/('core_metrics.json' if args.core_only else 'metrics.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['solver_calibration','q_calibration','p_calibration','trigger_calibration']},indent=2))

if __name__=='__main__':main()
