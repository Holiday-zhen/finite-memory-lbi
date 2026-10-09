"""Stable elastic-net Bregman energies and auditable reference solutions."""
import numpy as np
import mpmath as mp
from scipy.optimize import minimize, least_squares

def soft(s,tau):return np.sign(s)*np.maximum(np.abs(s)-tau,0.)

def direct_energy(t,x,s,tau):
    return float(tau*np.linalg.norm(t,1)+.5*(t@t)-tau*np.linalg.norm(x,1)-.5*(x@x)-s@(t-x))

def bregman(t,x,s,tau):
    # For x=soft(s,tau), z=clip(s,-tau,tau) is its l1 subgradient.
    # Algebraically D=.5||t-x||^2+sum((tau-z)t_+ +(tau+z)(-t)_+).
    # This avoids subtracting large nearly equal objective values.
    expected=soft(s,tau)
    if not np.allclose(x,expected,rtol=0,atol=16*np.finfo(float).eps*max(1.,float(np.max(np.abs(s))))):
        raise FloatingPointError('Primal-dual mirror consistency lost')
    tt=np.asarray(t,dtype=np.longdouble);xx=np.asarray(x,dtype=np.longdouble)
    zz=np.clip(np.asarray(s,dtype=np.longdouble),-np.longdouble(tau),np.longdouble(tau))
    raw=.5*np.sum((tt-xx)**2)+np.sum((tau-zz)*np.maximum(tt,0)+(tau+zz)*np.maximum(-tt,0))
    scale=max(1.,float(np.sum(tt*tt)+np.sum(xx*xx)))
    tolerance=64*np.finfo(float).eps*scale
    if raw < -tolerance:raise FloatingPointError(f'Negative raw Bregman energy {raw}')
    direct=direct_energy(np.asarray(t),x,s,tau)
    direct_tol=128*np.finfo(float).eps*max(1.,float(tau*(np.linalg.norm(t,1)+np.linalg.norm(x,1))+.5*(np.asarray(t)@np.asarray(t)+x@x)+abs(s@(np.asarray(t)-x))))
    if direct < -direct_tol:raise FloatingPointError(f'Negative direct Bregman energy {direct}')
    return float(max(raw,0))

def matrix(a):return mp.matrix([[mp.mpf(float(v)) for v in row] for row in a])

def reference(A,b,tau,tolerance):
    U,sv,Vt=np.linalg.svd(A,full_matrices=False)
    rank=int(np.sum(sv>1e-11*max(sv[0],1.)))
    R=Vt[:rank];c=(U[:,:rank].T@b)/sv[:rank]
    def fun(y):
        x=soft(R.T@y,tau)
        return .5*float(x@x)-float(c@y),R@x-c
    initial=minimize(fun,np.zeros(rank),jac=True,method='L-BFGS-B',options=dict(maxiter=5000,gtol=1e-12,ftol=1e-15,maxls=80))
    def jac(y):
        Q=R[:,np.abs(R.T@y)>tau]
        return Q@Q.T
    refine=least_squares(lambda y:fun(y)[1],initial.x,jac=jac,xtol=max(tolerance*.01,3e-15),ftol=max(tolerance*.01,3e-15),gtol=max(tolerance*.01,3e-15),max_nfev=3000)
    y=refine.x;s=R.T@y;x=soft(s,tau)
    primal=float(np.linalg.norm(R@x-c))
    if not refine.success or primal>tolerance:raise RuntimeError(f'Reference failed {primal:.3e} > {tolerance:.3e}')
    # A feasible primal point and an unconstrained dual point certify a gap
    # for the stored compact linear constraints, evaluated at 60 digits.
    with mp.workdps(60):
        RM=matrix(R);cm=mp.matrix([mp.mpf(float(v)) for v in c]);ym=mp.matrix([mp.mpf(float(v)) for v in y])
        xm=mp.matrix([mp.mpf(float(v)) for v in x]);sm=RM.T*ym;tm=mp.mpf(float(tau))
        feasible=xm+RM.T*mp.lu_solve(RM*RM.T,cm-RM*xm)
        omega=lambda v:tm*mp.fsum(abs(z) for z in v)+mp.fsum(z*z for z in v)/2
        star=mp.fsum(max(abs(z)-tm,0)**2 for z in sm)/2
        gap=omega(feasible)+star-mp.fsum(cm[i]*ym[i] for i in range(rank))
        if gap < -mp.mpf('1e-45'):raise FloatingPointError(f'Negative certified gap {gap}')
        bound=mp.sqrt(2*max(gap,0))+mp.norm(xm-feasible)
        diag=dict(acceptance_tolerance=tolerance,primal_feasibility=primal,
                  normal_residual=float(np.linalg.norm(A.T@(A@x-b))/np.linalg.norm(A.T@b)),
                  kkt_residual=float(np.linalg.norm(x-soft(s,tau))),
                  primal_dual_gap=float(gap),reference_distance_bound=float(bound),
                  lbfgsb_success=bool(initial.success),refine_success=bool(refine.success),
                  refiner_status=int(refine.status),nfev=int(refine.nfev),gap_digits=60)
    return x,diag

def test():
    rng=np.random.default_rng(17)
    for _ in range(20):
        s=rng.normal(size=20);x=soft(s,.2);t=rng.normal(size=20)
        assert np.isclose(bregman(t,x,s,.2),direct_energy(t,x,s,.2),rtol=1e-13)
        assert bregman(x,x,s,.2)==0
    assert bregman(np.array([1.+1e-10]),np.array([1.]),np.array([1.2]),.2)>0
    print('PASS stable Bregman identities and near-coincident points',flush=True)

if __name__=='__main__':test()
