"""Reproducible local search, with all function evaluations and seeds recorded."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from core import SymmetricDepolarizing, load_payload

ROOT=Path(__file__).resolve().parent

def optimize_state(psi,p,output,maxiter=300,real=False,mask=None,seed=20261011,cutoff=None):
    psi=np.asarray(psi,complex);psi/=np.linalg.norm(psi)
    n=psi.shape[1]-1;shape=psi.shape
    if mask is None:mask=np.ones(shape,bool)
    mask=np.asarray(mask,bool)
    model=SymmetricDepolarizing(n,p,cutoff=cutoff,cache_dir=ROOT/'cache')
    if np.any(mask[:,model.cutoff+1:]):
        raise ValueError('Optimization mask must be zero beyond the input cutoff.')
    output=Path(output);output.parent.mkdir(exist_ok=True,parents=True)
    vals=psi[mask]
    x0=vals.real.copy() if real else np.concatenate([vals.real,vals.imag])
    count=len(vals);history=[];start=time.perf_counter();best=-np.inf;best_state=psi.copy()
    scale=1e4
    def unpack(x):
        a=np.zeros(shape,complex)
        a[mask]=x if real else x[:count]+1j*x[count:]
        return a
    def objective(x):
        nonlocal best,best_state
        a=unpack(x);d=model.evaluate(a,gradient=True)
        g_full=np.zeros(shape,complex)
        g_full[:,:model.cutoff+1]=d.pop('gradient')
        g=g_full[mask]
        d.update(evaluation=len(history)+1,elapsed_seconds=time.perf_counter()-start)
        history.append(d)
        if d['ic_bits']>best:
            best=d['ic_bits'];best_state=a/np.linalg.norm(a)
            np.savez(output.with_suffix('.npz'),psi=best_state,p_total=p,n=n,ic_bits=best)
        if len(history)%25==0:
            print(json.dumps({'eval':len(history),'best_ic':best,'current_ic':d['ic_bits'],'elapsed':d['elapsed_seconds']}),flush=True)
            output.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
        packed=g.real if real else np.concatenate([g.real,g.imag])
        return -scale*d['ic_bits'],-scale*packed
    result=minimize(objective,x0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-14,'gtol':1e-9,'maxls':30,'maxcor':20})
    final=model.evaluate(best_state,diagnostics=True)
    report={k:(v.item() if isinstance(v,np.generic) else v) for k,v in final.items()}
    report.update(seed=seed,method='L-BFGS-B',parameterization='real masked' if real else 'complex masked',
        maxiter=maxiter,iterations=int(result.nit),evaluations=int(result.nfev),optimizer_success=bool(result.success),
        optimizer_message=str(result.message),elapsed_seconds=time.perf_counter()-start,initial_ic=history[0]['ic_bits'],
        objective_scale=scale,gradient_regularization='log eigenvalues floored at sector trace times1e-15',
        evidence='floating point; not a rigorous certificate')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    output.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
    best_state.astype('<c16').tofile(output.with_suffix('.c128'))
    print(json.dumps(report),flush=True)
    return best_state,report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--p',type=float,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--maxiter',type=int,default=300)
    ap.add_argument('--real',action='store_true');ap.add_argument('--n',type=int);ap.add_argument('--kick',type=float,default=0.)
    ap.add_argument('--seed',type=int,default=20261011);args=ap.parse_args()
    path=Path(args.input)
    psi=load_payload(path) if path.suffix=='.c128' else np.load(path)['psi']
    if args.n and args.n!=psi.shape[1]-1:
        if args.n<psi.shape[1]-1:raise ValueError('Truncation must be explicit, not silently applied.')
        psi=np.pad(psi,((0,0),(0,args.n+1-psi.shape[1])))
    if args.kick:
        rng=np.random.default_rng(args.seed)
        psi=psi+args.kick*(rng.normal(size=psi.shape)+1j*rng.normal(size=psi.shape))/np.sqrt(psi.size)
    optimize_state(psi,args.p,args.output,args.maxiter,args.real,seed=args.seed)
