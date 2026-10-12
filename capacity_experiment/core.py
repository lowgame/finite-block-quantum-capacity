"""Finite-block depolarizing coherent information on symmetric inputs.

Independent implementation of Appendix E, arXiv:2608.15870v2.
Public API uses TOTAL Pauli error p: D_p=(1-p)rho+p/3 sum_P P rho P.
Input psi has shape (reference rank, n+1), coefficients in normalized Dicke basis.
Every spin sector is included. Kernels incorporate representation multiplicities.
Floating-point evaluation and optimization are NOT rigorous certificates.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math, time
import numpy as np
from scipy.linalg import eigh


@dataclass
class Sector:
    q: int
    d: int
    multiplicity: int
    diagonals: list[np.ndarray]


def _entropy(values):
    good=values[values>0]
    return float(-np.dot(good,np.log2(good)))


class SymmetricDepolarizing:
    def __init__(self,n:int,p:float,cutoff:int|None=None,cache_dir=None):
        if n<1 or not 0<=p<.75:
            raise ValueError('Require n>=1 and 0<=total Pauli p<0.75.')
        self.n=int(n);self.p=float(p)
        self.cutoff=self.n if cutoff is None else min(int(cutoff),self.n)
        self.sectors=[]
        self.build_seconds=0.
        self._cache=Path(cache_dir) if cache_dir else None
        start=time.perf_counter()
        if self._cache:
            self._cache.mkdir(parents=True,exist_ok=True)
            path=self._cache/f'kernel_n{n}_k{self.cutoff}_p{self.p.hex()}.npz'
            if path.exists():
                z=np.load(path,allow_pickle=False)
                for q in range(n//2+1):
                    d=n-2*q;m=math.comb(n,q)-(math.comb(n,q-1) if q else 0)
                    di=[z[f'q{q}_a{a}'] for a in range(min(d,self.cutoff)+1)]
                    self.sectors.append(Sector(q,d,m,di))
                self.build_seconds=time.perf_counter()-start
                return
        self._build()
        self.build_seconds=time.perf_counter()-start
        if self._cache:
            values={f'q{s.q}_a{a}':v for s in self.sectors for a,v in enumerate(s.diagonals)}
            np.savez_compressed(path,**values)

    def _build(self):
        n=self.n;K=self.cutoff;r=self.p/3
        a=1-2*r;b=2*r;eta=1-4*r;det=2*r*(1-2*r)
        choose=np.zeros((n+1,n+1),dtype=float)
        for j in range(n+1):
            for k in range(j+1):choose[j,k]=math.comb(j,k)
        ap=[np.array([math.comb(s,k)*a**(s-k)*b**k for k in range(s+1)]) for s in range(n+1)]
        dp=[np.array([math.comb(e,k)*b**(e-k)*a**k for k in range(e+1)]) for e in range(n+1)]
        ad={}
        for q in range(n//2+1):
            d=n-2*q
            mult=math.comb(n,q)-(math.comb(n,q-1) if q else 0)
            detpoly=np.array([math.comb(2*q,h)*det**q*mult for h in range(2*q+1)])
            polys={};di=[]
            for delta in range(min(d,K)+1):
                kval=np.arange(delta,K+1)
                invnorm=1/np.sqrt(choose[n,kval]*choose[n,kval-delta])
                kernel=np.zeros((d+1-delta,len(kval)))
                for j in range(d+1-delta):
                    i=j+delta;u=d-j;v=d-i
                    coeff=np.zeros(n+1)
                    for s in range(max(0,u+v-d),min(u,v)+1):
                        bb=v-s;cc=u-s;e=d-u-v+s
                        pair=(s,e)
                        if pair not in ad:ad[pair]=np.convolve(ap[s],dp[e])
                        if pair not in polys:polys[pair]=np.convolve(ad[pair],detpoly)
                        pol=polys[pair]
                        factor=choose[v,s]*choose[d-v,u-s]*eta**(bb+cc)
                        coeff[cc:cc+len(pol)]+=factor*pol
                    kernel[j,:]=coeff[kval]*math.sqrt(choose[d,i]/choose[d,j])*invnorm
                di.append(kernel)
            self.sectors.append(Sector(q,d,mult,di))

    def blocks(self,psi,weighted=True):
        psi=np.asarray(psi,dtype=complex)
        if psi.ndim!=2 or psi.shape[1] not in (self.cutoff+1,self.n+1):
            raise ValueError('psi shape must be (rank,cutoff+1) or (rank,n+1).')
        if psi.shape[1]>self.cutoff+1:
            if np.any(psi[:,self.cutoff+1:]!=0):raise ValueError('Nonzero amplitudes beyond cutoff.')
            psi=psi[:,:self.cutoff+1]
        norm=float(np.vdot(psi,psi).real)
        if norm<=0:raise ValueError('Zero state.')
        psi=psi/math.sqrt(norm);rank=len(psi);out=[]
        for sec in self.sectors:
            D=sec.d+1;B=np.zeros((rank,rank,D,D),dtype=complex)
            for delta,ker in enumerate(sec.diagonals):
                hi=np.arange(delta,self.cutoff+1);lo=hi-delta
                vals=np.einsum('rk,ck,ik->rci',psi[:,hi],psi[:,lo].conj(),ker,optimize=True)
                jj=np.arange(D-delta);ii=jj+delta
                B[:,:,ii,jj]=vals
                if delta:B[:,:,jj,ii]=vals.transpose(1,0,2).conj()
            if not weighted:B/=sec.multiplicity
            out.append(B.transpose(0,2,1,3).reshape(rank*D,rank*D))
        return out

    def evaluate(self,psi,gradient=False,diagnostics=False):
        x=np.asarray(psi,dtype=complex)
        if x.ndim!=2 or x.shape[1] not in (self.cutoff+1,self.n+1):
            raise ValueError('psi shape must be (rank,cutoff+1) or (rank,n+1).')
        if not np.all(np.isfinite(x)) or not len(x):
            raise ValueError('Finite nonempty coefficients are required.')
        if np.any(x[:,self.cutoff+1:]!=0):
            raise ValueError('Nonzero amplitudes beyond cutoff.')
        x=x[:,:self.cutoff+1]
        norm=float(np.vdot(x,x).real)
        if norm<=0:raise ValueError('Zero state.')
        v=x/math.sqrt(norm);rank=len(x)
        blocks=self.blocks(v);ic=0.;trace=0.;minimum=0.;negative_mass=0.
        hv=np.zeros_like(v) if gradient else None
        sector_values=[]
        for sec,B in zip(self.sectors,blocks):
            D=sec.d+1
            B=(B+B.conj().T)*.5
            U=B.reshape(rank,D,rank,D).transpose(0,2,1,3)
            O=np.einsum('rrij->ij',U)
            O=(O+O.conj().T)*.5
            if gradient:
                wb,Vb=eigh(B,check_finite=False,driver='evd')
                wo,Vo=eigh(O,check_finite=False,driver='evd')
            else:
                wb=eigh(B,eigvals_only=True,check_finite=False,driver='evr')
                wo=eigh(O,eigvals_only=True,check_finite=False,driver='evr')
            weight=float(np.trace(O).real);trace+=weight
            minimum=min(minimum,float(wb.min()),float(wo.min()))
            negative_mass+=float(-wb[wb<0].sum()-wo[wo<0].sum())
            val=_entropy(wo)-_entropy(wb);ic+=val
            sector_values.append({'q':sec.q,'d':sec.d,'weight':weight,'ic_bits':val})
            if gradient:
                # Derivative of entropy difference; identity terms cancel.
                floor=max(abs(weight)*1e-15,1e-300)
                Lb=(Vb*np.log2(np.maximum(wb,floor)))@Vb.conj().T
                Lo=(Vo*np.log2(np.maximum(wo,floor)))@Vo.conj().T
                G=Lb.reshape(rank,D,rank,D).transpose(0,2,1,3).copy()
                for rr in range(rank):G[rr,rr]-=Lo
                for delta,ker in enumerate(sec.diagonals):
                    hi=np.arange(delta,self.cutoff+1);lo=hi-delta
                    jj=np.arange(D-delta);ii=jj+delta
                    H=np.einsum('rci,ik->rck',G[:,:,ii,jj],ker,optimize=True)
                    hv[:,hi]+=np.einsum('rck,ck->rk',H,v[:,lo],optimize=True)
                    if delta:hv[:,lo]+=np.einsum('rck,rk->ck',H.conj(),v[:,hi],optimize=True)
        grad=2*(hv-ic*v)/math.sqrt(norm) if gradient else None
        result={'ic_bits':float(ic),'rate':float(ic/self.n),'trace':trace,
                'min_eigenvalue':minimum,'negative_eigenvalue_mass':negative_mass,
                'n':self.n,'p_total':self.p,'cutoff':self.cutoff}
        if diagnostics:result['sectors']=sector_values
        if gradient:result['gradient']=grad
        return result


def load_payload(path,n=45):
    data=np.fromfile(path,dtype='<c16')
    if data.size!=2*(n+1):raise ValueError('Unexpected payload size.')
    return data.reshape(2,n+1)


def repetition(n):
    a=np.zeros((2,n+1),complex);a[0,0]=1/math.sqrt(2);a[1,n]=1/math.sqrt(2)
    return a


if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=5)
    ap.add_argument('--p',type=float,default=.19);ap.add_argument('--payload')
    args=ap.parse_args();model=SymmetricDepolarizing(args.n,args.p)
    psi=load_payload(args.payload,args.n) if args.payload else repetition(args.n)
    print(json.dumps({'build_seconds':model.build_seconds,**model.evaluate(psi,diagnostics=True)},indent=2))
