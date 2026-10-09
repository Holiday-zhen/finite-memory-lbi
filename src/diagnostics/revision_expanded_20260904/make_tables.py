"""Mechanically format measured results; never supply fabricated placeholders."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'deliverable/tables'

def sci(x,d=3):
    x=float(x)
    if x==0:return '0'
    a,b=f'{x:.{d-1}e}'.split('e')
    return a+r'\times10^{'+str(int(b))+'}'

def table(file,caption,label,fmt,header,rows):
    text='\\begin{table}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+fmt+'}\n\\hline\n'
    text+=' & '.join(header)+r'\\'+'\n\\hline\n'
    text+='\n'.join(' & '.join(row)+r'\\' for row in rows)
    text+='\n\\hline\n\\end{tabular}\n\\end{table}\n'
    (OUT/file).write_text(text,encoding='utf-8')

def csvrows(name):return list(csv.DictReader((HERE/'analysis'/name).open(encoding='utf-8-sig')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--core-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    m=json.loads((HERE/'analysis'/('core_metrics.json' if args.core_only else 'metrics.json')).read_text())
    mp=json.loads((HERE/'analysis/reference_mp60.json').read_text())
    e=csvrows('energy_per_instance.csv');refs=[r for r in csvrows('reference_accuracy.csv') if r['study']=='energy']
    values=dict(EnergyMaxPG=max(float(r['actual_max_pgn']) for r in e),ReferenceNormal=max(float(r['normal_residual']) for r in refs),MPReferenceChange=mp['max_relative_energy_change'],MPFactorChange=mp['factor_change'],MinimumRawEnergy=m['energy_accuracy']['min_raw_direct'],GainMargin=m['gain']['min_margin'])
    if not args.core_only:values['HardDisplayFloor']=m['hard']['zero_display_floor']
    (OUT/'values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{$'+sci(v)+'$}' for k,v in values.items())+'\n',encoding='utf-8')
    names={'lbfgsb':'L-BFGS-B','pg':'Projected gradient','fista':'Accelerated PG','bb':'Projected BB'}
    table('calibration.tex','Inner-solver calibration: means over 10 instances.','tab:solver-calibration','lrrr',['Solver','Residual at 70','Total time (s)','$t(10^{-3})$ (s)'],[[names[r['solver']],'$'+sci(r['final_residual'])+'$',f"{float(r['runtime']):.4f}",f"{float(r['time_to_1e-3']):.4f}"] for r in m['solver_calibration']])
    table('energy.tex','Energy at iteration 75 and fitted factors over iterations 40--75 (20 instances).','tab:energy','lrrr',['Method','Median energy ($10^{-9}$)','Energy IQR ($10^{-9}$)','Mean $\\rho$'],[[r['method'],f"{r['final']['median']*1e9:.2f}",f"[{r['final']['q25']*1e9:.2f}, {r['final']['q75']*1e9:.2f}]",f"{r['factor']['mean']:.4f}"] for r in m['energy']])
    names={'fixed':'Fixed tolerance','certified':'Certified','accurate':'High accuracy'}
    table('certificate.tex','Means over 20 instances; inner iterations are per outer step, violations per 80 steps.','tab:certificate-cost','lrrrr',['Stopping rule','Final energy','Time (s)','Inner it.','Violations'],[[names[r['rule']],'$'+sci(r['final_energy']['mean'])+'$',f"{r['runtime']['mean']:.3f}",f"{r['mean_inner']['mean']:.2f}",f"{r['budget_violations']['mean']:.1f}"] for r in m['certificate']])
    scale=csvrows('scale_per_instance.csv');rows=[]
    for name in ['one-cut','recent','violated','angle','adaptive']:
        rs=[r for r in scale if r['n']=='12000' and r['method']==name];v=np.array([float(r['final_residual']) for r in rs])*1e4;t=np.array([float(r['runtime']) for r in rs])
        rows.append([name,f'{np.median(v):.2f}',f'[{np.quantile(v,.25):.2f}, {np.quantile(v,.75):.2f}]',f'{np.median(t):.3f}',str(sum(int(r['success']) for r in rs))+'/20'])
    table('scale.tex','Largest sparse size ($n=12000$): residuals, median times, and target successes.','tab:scale','lrrrr',['Method','Median ($10^{-4}$)','IQR ($10^{-4}$)','Time (s)','Success'],rows)
    if args.core_only:return
    hard=[r for r in m['hard']['rows'] if r['cohort']=='all'];rows=[]
    for p,label in [('column_block','Column-block'),('binary_digits','Binary Digits')]:
        for j,r in enumerate(x for x in hard if x['problem']==p):
            rows.append([label if j==0 else '',r['method'],'$'+sci(r['median'])+'$','$['+sci(r['q25'])+r',\ '+sci(r['q75'])+']$',f"{r['lower']}/20" if r['method']!='one-cut' else '--'])
    table('hard.tex','Terminal residuals over 20 instances; lower counts compare with one cut on the same instance.','tab:hard-problems','llrrr',['Problem','Method','Median','IQR','Lower'],rows)
    p=HERE/'data/hard_mp';previous=HERE.parent/'extended_horizon_20260904/mp_results';ar=[]
    for seed in range(20,40):
        for name in ['one-cut','recent','violated','angle','adaptive']:
            r=json.loads(((previous if seed<30 else p)/f'binary_digits__{seed}__{name}__dps50.json').read_text());ar.append(r)
    error=max(abs(float(r['mp_saved_x_residual'])-r['recorded_residual'])/max(abs(r['recorded_residual']),1e-300) for r in ar)
    mp_summary=[]
    for name in ['one-cut','recent','violated','angle','adaptive']:
        rs=[r for r in ar if r['method']==name];base={r['seed']:float(r['mp_mirror_residual']) for r in ar if r['method']=='one-cut'}
        mp_summary.append(dict(method=name,median_saved=float(np.median([float(r['mp_saved_x_residual']) for r in rs])),median_remirror=float(np.median([float(r['mp_mirror_residual']) for r in rs])),remirror_lower=sum(float(r['mp_mirror_residual'])<base[r['seed']] for r in rs)))
    with (HERE/'analysis/hard_mp50_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(mp_summary[0]));w.writeheader();w.writerows(mp_summary)
    (HERE/'analysis/hard_mp50_accuracy.json').write_text(json.dumps(dict(max_saved_relative_difference=error,summary=mp_summary),indent=2),encoding='utf-8')
    print('Hard table and audit summary ready. Write the interpretation from these measured values:',hard,mp_summary)

if __name__=='__main__':main()
