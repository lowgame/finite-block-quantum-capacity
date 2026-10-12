"""Deterministic Gaussian-comb screen at one finite block size."""
import argparse,json,time
from pathlib import Path
import numpy as np
from core import SymmetricDepolarizing
from candidates.families import gaussian_comb

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,required=True);ap.add_argument('--p',type=float,default=.197)
args=ap.parse_args();n=args.n;K=min(n,72)
start=time.perf_counter();m=SymmetricDepolarizing(n,args.p,cutoff=K,cache_dir=ROOT/'cache')
records=[];best=-np.inf
for mu in [16.,24.,32.,40.]:
 for sd in [6.,9.,12.]:
  for q in [.4,.5]:
   a=gaussian_comb(n,K,means=[mu,mu],sigmas=[sd,sd],weights=[q,1-q])
   d=m.evaluate(a);d.update(mean=mu,sigma=sd,q=q,seconds=time.perf_counter()-start)
   records.append(d)
   if d['ic_bits']>best:
    best=d['ic_bits'];np.savez(ROOT/f'results/gaussian_screen_n{n}.npz',psi=a,n=n,p_total=args.p,ic_bits=best,mean=mu,sigma=sd,q=q)
    print(json.dumps(d),flush=True)
(ROOT/f'results/gaussian_screen_n{n}.json').write_text(json.dumps(records,indent=2))
print('COMPLETE',n,len(records),'best',best,'seconds',time.perf_counter()-start,flush=True)
