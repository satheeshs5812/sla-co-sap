"""Paper 2 statistics, revised: Quade test (instance-difficulty weighted) + exact sign tests with Finner
correction for P2 vs each method, and Dolan-More performance profiles. Replaces Friedman/Nemenyi/Wilcoxon-Holm."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata, f as fdist, binomtest
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS as R  # <repo>/results (override with SLACO_RESULTS)
COL = {"P1-SLA-CO-MILP": "#0072B2", "P2-SLA-ALNS": "#D55E00", "B1-Industry": "#000000", "B2-Det-MILP": "#56B4E9","B3-GA": "#009E73", "B4-SA": "#CC79A7"}
LS = {"P1-SLA-CO-MILP": "-", "P2-SLA-ALNS": "-", "B1-Industry": ":", "B2-Det-MILP": "--","B3-GA": "-.", "B4-SA": (0,(5,1,1,1))}
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
d=pd.read_csv(f'{R}/ALL_main_per_instance.csv')
M=['P1-SLA-CO-MILP','P2-SLA-ALNS','B1-Industry','B2-Det-MILP','B3-GA','B4-SA']
def quade(X):  # X: b x k, lower is better
    b,k=X.shape; rng=X.max(1)-X.min(1); Q=rankdata(rng); r=np.apply_along_axis(rankdata,1,X)
    S=Q[:,None]*(r-(k+1)/2); A=(S**2).sum(); B=(S.sum(0)**2).sum()/b
    F=(b-1)*B/(A-B) if A>B else np.inf; p=fdist.sf(F,k-1,(b-1)*(k-1)); return F,p,(Q[:,None]*r).sum(0)/(b*(b+1)/2)
def finner(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); adj=np.empty(m); cur=0
    for i,j in enumerate(o):
        cur=max(cur,1-(1-p[j])**(m/(i+1))); adj[j]=min(cur,1)
    return adj
out=[];pair=[]
for met in ['sim_cost_mean','sim_prd_ev_viol_rate','sim_cvar95']:
    P=d.pivot(index='instance',columns='method',values=met)[M]
    fam=d.drop_duplicates('instance').set_index('instance').family.reindex(P.index)
    for fm in ['ALL','S','M','L']:
        X=P.values if fm=='ALL' else P[fam==fm].values
        F,p,w=quade(X); out.append(dict(metric=met,family=fm,b=len(X),F=F,df1=X.shape[1]-1,df2=(len(X)-1)*(X.shape[1]-1),p=p,**{f'wrank_{m}':v for m,v in zip(M,w)},**{f'rank_{m}':v for m,v in zip(M,np.apply_along_axis(rankdata,1,X).mean(0))}))
    rows=[]
    for m in M:
        if m=='P2-SLA-ALNS': continue
        x=P['P2-SLA-ALNS'].values; y=P[m].values; tol=1e-9*np.maximum(1,np.abs(y))
        w=int((x<y-tol).sum()); l=int((x>y+tol).sum()); t=len(x)-w-l
        rows.append(dict(metric=met,comparison=f'P2 vs {m}',W=w,T=t,L=l,p=binomtest(w,w+l,0.5).pvalue if w+l else 1.0))
    adj=finner([r['p'] for r in rows])
    for r,a in zip(rows,adj): r['p_finner']=a
    pair+=rows
pd.DataFrame(out).to_csv(f'{R}/ALL_quade.csv',index=False); pd.DataFrame(pair).to_csv(f'{R}/ALL_sign_finner.csv',index=False)
print(pd.DataFrame(out)[['metric','family','b','F','p']].to_string()); print(pd.DataFrame(pair).to_string())
# ablation: exact sign test per variant vs full (paired by instance, seed)
a=pd.read_csv(f'{R}/ALL_ablation.csv'); full=a[a.variant=='full'].set_index(['instance','seed']).sim_cost_mean
for v in a.variant.unique():
    if v=='full': continue
    y=a[a.variant==v].set_index(['instance','seed']).sim_cost_mean.reindex(full.index)
    w=int((full<y-1e-6).sum()); l=int((full>y+1e-6).sum())
    print('ablation',v,'full better',w,'worse',l,'ties',len(y)-w-l,'sign p',round(binomtest(w,w+l,.5).pvalue,3) if w+l else 1)
# Dolan-More performance profiles
fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,met,tt in zip(axs,['sim_cost_mean','sim_cvar95'],['(a) realised mean cost','(b) CVaR$_{95}$ of realised cost']):
    P=d.pivot(index='instance',columns='method',values=met)[M]; ratio=P.div(P.min(1),axis=0)
    taus=np.linspace(1,ratio.values.max()*1.001,2000)
    for m in M:
        rho=[(ratio[m]<=t).mean() for t in taus]
        ax.step(taus,rho,where='post',color=COL[m],ls=LS[m],lw=1.6 if m.startswith('P2') else 1.2,label=m)
    ax.set_xscale('log'); ax.set_xlim(1,taus.max()); ax.set_ylim(0,1.02)
    ax.set_xlabel(r'performance ratio $\tau$ (cost / best cost on the instance)'); ax.set_title(tt,loc='left')
    ax.grid(alpha=0.25,lw=0.5)
    from matplotlib.ticker import FixedLocator,FormatStrFormatter
    tk=[1,1.01,1.02,1.05,1.1]; tk=[t for t in tk if t<=taus.max()]; ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
axs[0].set_ylabel(r'share of instances $\rho(\tau)$'); axs[1].legend(loc='lower right',frameon=False)
fig.tight_layout(); fig.savefig(f'{R}/figures/fig9_performance_profiles.png'); fig.savefig(f'{R}/figures/fig9_performance_profiles.pdf')
P=d.pivot(index='instance',columns='method',values='sim_cost_mean')[M]; ratio=P.div(P.min(1),axis=0)
print('rho(1):',(ratio<=1+1e-12).mean().round(3).to_dict()); print('rho(1.01):',(ratio<=1.01).mean().round(3).to_dict()); print('max ratio',ratio.max().round(4).to_dict())
P=d.pivot(index='instance',columns='method',values='sim_cvar95')[M]; ratio=P.div(P.min(1),axis=0)
print('CVaR rho(1):',(ratio<=1+1e-12).mean().round(3).to_dict()); print('CVaR rho(1.01):',(ratio<=1.01).mean().round(3).to_dict())
# Quade post-hoc (Conover, 1999): pairwise t on summed weighted centred ranks, Finner-adjusted
from scipy.stats import t as tdist
qp=[]
for met in ['sim_cost_mean','sim_prd_ev_viol_rate','sim_cvar95']:
    P=d.pivot(index='instance',columns='method',values=met)[M]; X=P.values; b,k=X.shape
    Q=rankdata(X.max(1)-X.min(1)); r=np.apply_along_axis(rankdata,1,X); S=Q[:,None]*(r-(k+1)/2)
    A=(S**2).sum(); B=(S.sum(0)**2).sum()/b; Sj=S.sum(0); se=np.sqrt(2*b*(A-B)/((b-1)*(k-1))); df=(b-1)*(k-1)
    i2=M.index('P2-SLA-ALNS'); rows=[]
    for j,m in enumerate(M):
        if j==i2: continue
        tt=(Sj[i2]-Sj[j])/se; rows.append(dict(metric=met,comparison=f'P2 vs {m}',t=tt,p=2*tdist.sf(abs(tt),df)))
    adj=finner([r_['p'] for r_ in rows])
    for r_,a_ in zip(rows,adj): r_['p_finner']=a_
    qp+=rows
qp=pd.DataFrame(qp); qp.to_csv(f'{R}/ALL_quade_posthoc_finner.csv',index=False); print(qp.to_string())
