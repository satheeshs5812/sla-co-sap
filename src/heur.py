"""Heuristics: B1 industry wave heuristic, P2 SLA-ALNS, B3 GA, B4 SA."""
import time, math
import numpy as np
from calib import *
from model import solve_cpsat, BIG_UNSCHED

LAM = 1e6  # penalty per unit violation for penalty-based metaheuristics


class State:
    def __init__(self, P):
        self.P = P
        T1 = P.T + 1
        self.sel = np.full(P.nJ, -1)
        self.eff = np.zeros(T1); self.trf = np.zeros(T1); self.par = np.zeros(T1)
        self.busy = np.zeros((P.nS, T1), bool)
        self.md = np.zeros((P.nS, P.nM), int)
        self.tau = np.zeros(P.nS, int)
        self.hsel = P.best_h(np.ones(P.nS, int))

    def copy(self):
        o = State.__new__(State)
        o.P = self.P
        for a in ("sel", "eff", "trf", "par", "busy", "md", "tau", "hsel"):
            setattr(o, a, getattr(self, a).copy())
        return o

    def insert(self, j, g):
        P = self.P; t = P.O_t[g]; i = P.O_sys[g]
        self.sel[j] = g; self.eff[t] += P.O_eff[g]; self.trf[t] += P.O_trf[g]
        self.busy[i, t] = True; self.md[i, P.O_month[g]] += P.O_qmin[g]
        if P.O_trans[g]:
            self.par[t] += 1; self.tau[i] = t

    def remove(self, j):
        P = self.P; g = self.sel[j]
        if g < 0:
            return
        t = P.O_t[g]; i = P.O_sys[g]
        self.sel[j] = -1; self.eff[t] -= P.O_eff[g]; self.trf[t] -= P.O_trf[g]
        self.busy[i, t] = False; self.md[i, P.O_month[g]] -= P.O_qmin[g]
        if P.O_trans[g]:
            self.par[t] -= 1; self.tau[i] = 0

    def candidates(self, j, cand=None):
        P = self.P
        g = P.allowed[j] if cand is None else cand
        if len(g) == 0:
            return g, np.zeros(0)
        t = P.O_t[g]; i = P.job_sys[j]
        ok = ~self.busy[i, t]
        ok &= self.eff[t] + P.O_eff[g] <= P.C[t] + 1e-6
        ok &= self.trf[t] + P.O_trf[g] <= P.Bcap[t] + 1e-6
        ok &= self.md[i, P.O_month[g]] + P.O_qmin[g] <= P.budget_min[i]
        cost = P.O_cost[g].copy()
        if P.job_trans[j]:
            ok &= self.par[t] + 1 <= P.Pmax[t]
            a = P.pred[i]
            if a is not None and self.tau[a] > 0:
                ok &= t >= self.tau[a] + P.lag[i]
            b = P.succ[i]
            if b is not None and self.tau[b] > 0:
                ok &= t <= self.tau[b] - P.lag[b]
            if P.migr[i]:
                cw = P.cloud_week[i, self.hsel[P.comp[i]]]
                if not np.isfinite(cw):
                    cw = np.min(P.cloud_week[i])
                cost += cw * (P.T - t)
            for nb, pen, kind in P.nbrs[i]:
                if self.tau[nb] > 0:
                    if P.hard and kind == "tight":
                        ok &= t == self.tau[nb]
                    else:
                        cost += pen * np.abs(t - self.tau[nb])
        return g[ok], cost[ok]

    def refresh_h(self):
        self.hsel = self.P.best_h(self.tau)


# ------------------------------------------------------------------ repairs
def greedy_repair(st, jobs, rng):
    P = st.P
    jobs = list(jobs)
    rng.shuffle(jobs)
    jobs.sort(key=lambda j: len(P.allowed[j]))
    left = []
    for j in jobs:
        g, c = st.candidates(j)
        if len(g):
            st.insert(j, g[int(np.argmin(c))])
        else:
            left.append(j)
    return left


def regret_repair(st, jobs, rng):
    rem = list(jobs)
    left = []
    while rem:
        best = None
        for j in rem:
            g, c = st.candidates(j)
            if len(g) == 0:
                r = 1e18; gb = -1
            else:
                k = np.argsort(c)[:2]
                gb = g[k[0]]
                r = (c[k[1]] - c[k[0]]) if len(k) > 1 else 1e15
            if best is None or r > best[0]:
                best = (r, j, gb)
        _, j, gb = best
        rem.remove(j)
        if gb >= 0:
            st.insert(j, gb)
        else:
            left.append(j)
    return left


def cpsat_repair(st, jobs, rng, tl=2.0):
    P = st.P
    jobs = list(jobs)
    jset = set(jobs)
    allowed = [P.allowed[j] if (j in jset or st.sel[j] < 0) else np.array([st.sel[j]]) for j in range(P.nJ)]
    free_c = set(P.comp[P.job_sys[j]] for j in jobs if P.job_trans[j])
    fixed_h = {c: int(st.hsel[c]) for c in range(P.nC) if c not in free_c}
    r = solve_cpsat(P, time_limit=tl, workers=1, seed=int(rng.integers(1 << 30)), allowed=allowed, fixed_h=fixed_h)
    if r["sel"] is None:
        return greedy_repair(st, jobs, rng)
    for j in jobs:
        st.insert(j, r["sel"][j])
    return []


# ------------------------------------------------------------------ destroys
def d_random(st, k, rng, ctx):
    sch = np.nonzero(st.sel >= 0)[0]
    return list(rng.choice(sch, size=min(k, len(sch)), replace=False))


def d_cluster(st, k, rng, ctx):
    P = st.P
    seed = int(rng.integers(P.nS))
    seen = [seed]; q = [seed]; sset = {seed}
    land = ctx["land"]
    while q and sum(len(ctx["sys_jobs"][s]) for s in seen) < k:
        s = q.pop(0)
        for nb in [n for n, _, _ in P.nbrs[s]] + land[P.S[s]["landscape"]]:
            if nb not in sset:
                sset.add(nb); seen.append(nb); q.append(nb)
    out = [j for s in seen for j in ctx["sys_jobs"][s] if st.sel[j] >= 0]
    return out[:max(k, 3)]


def d_worst(st, k, rng, ctx):
    P = st.P
    sch = np.nonzero(st.sel >= 0)[0]
    excess = P.O_cost[st.sel[sch]] - ctx["minc"][sch]
    order = sch[np.argsort(-excess)]
    out = []
    lst = list(order)
    while len(out) < min(k, len(sch)):
        idx = int(len(lst) * rng.random() ** 3)
        out.append(lst.pop(idx))
    return out


def d_window(st, k, rng, ctx):
    P = st.P
    t = int(rng.integers(1, P.T + 1))
    sch = np.nonzero(st.sel >= 0)[0]
    tt = P.O_t[st.sel[sch]]
    out = list(sch[tt == t])
    if len(out) < k // 2:
        out += list(sch[tt == min(t + 1, P.T)])
    out = list(dict.fromkeys(out))
    if len(out) > k:
        out = list(rng.choice(out, size=k, replace=False))
    return out


def d_product(st, k, rng, ctx):
    P = st.P
    prods = ctx["prods"]
    p = prods[int(rng.integers(len(prods)))]
    out = [j for s in ctx["prod_sys"][p] for j in ctx["sys_jobs"][s] if st.sel[j] >= 0]
    if len(out) > k:
        out = list(rng.choice(out, size=k, replace=False))
    return out


DESTROYS = {"random": d_random, "cluster": d_cluster, "worst": d_worst, "window": d_window, "product": d_product}
REPAIRS = {"greedy": greedy_repair, "regret": regret_repair, "cpsat": cpsat_repair}


def make_ctx(P):
    sys_jobs = [[] for _ in range(P.nS)]
    for j in range(P.nJ):
        sys_jobs[P.job_sys[j]].append(j)
    land = {}
    for s in P.S:
        land.setdefault(s["landscape"], []).append(s["id"])
    prod_sys = {}
    for s in P.S:
        prod_sys.setdefault(s["product"], []).append(s["id"])
    minc = np.array([P.O_cost[P.allowed[j]].min() if len(P.allowed[j]) else 0 for j in range(P.nJ)])
    return dict(sys_jobs=sys_jobs, land=land, prod_sys=prod_sys, prods=list(prod_sys), minc=minc)


# ------------------------------------------------------------------ B1 industry heuristic
PROD_ORDER = ["SOLMAN", "PORTAL", "GRC", "SRM", "CRM", "PIPO", "BW", "S4", "ERP"]
DEFAULT = {"MIGUPG": "DMO_SM", "UPGRADE": "SUM_STD", "MAINT": "PATCH_STD"}


def industry(P):
    """Wave heuristic on the (deterministic) problem P: earliest feasible window, default method first,
    fall back to other methods by increasing service cost; cheapest SKU per co-location group."""
    st = State(P)
    st.hsel = P.best_h(np.ones(P.nS, int))
    order = []
    lands = sorted(set(s["landscape"] for s in P.S), key=lambda l: (PROD_ORDER.index([s for s in P.S if s["landscape"] == l][0]["product"]), l))
    tiers = ["DEV", "QAS", "PRD"]
    for tier in tiers:  # waves: all DEV, then QAS, then PRD, product order inside each wave
        for l in lands:
            for s in P.S:
                if s["landscape"] == l and s["tier"] == tier:
                    order.append(P.trans_job[s["id"]])
    order += [j for j in range(P.nJ) if not P.job_trans[j]]
    forced = 0
    for j in order:
        i = P.job_sys[j]; s = P.S[i]; jt = P.jobs[j]["jtype"]
        dm = DEFAULT.get(jt) or ("DMO_SM_MIG" if s["src"] == "anyDB" else "BR_HOMO")
        meths = sorted(set(P.O_m[P.allowed[j]]) | set(P.O_m[P.off[j]:P.end[j]]),
                       key=lambda m: (P.mnames[m] != dm, METHODS[P.mnames[m]]["service"]))
        placed = False
        for m in meths:
            cand = P.allowed[j][P.O_m[P.allowed[j]] == m]
            g, c = st.candidates(j, cand)
            if len(g):
                st.insert(j, g[np.argmin(P.O_t[g])]); placed = True; break
        if not placed:  # force earliest structurally possible option of the default method (plan infeasible)
            alls = np.arange(P.off[j], P.end[j])
            pick = alls[P.O_m[alls] == meths[0]]
            st.insert(j, pick[np.argmin(P.O_t[pick])]); forced += 1
    return st, forced


def state_from_plan(P, sel_plan, rng):
    """Insert jobs of a foreign plan into P keeping options that are feasible, greedily repairing the rest."""
    st = State(P)
    tau = np.zeros(P.nS, int)
    for j, g in enumerate(sel_plan):
        if g >= 0 and P.O_trans[g]:
            tau[P.O_sys[g]] = P.O_t[g]
    st.hsel = P.best_h(tau)
    rest = []
    order = sorted(range(P.nJ), key=lambda j: (not P.job_trans[j], P.O_t[sel_plan[j]] if sel_plan[j] >= 0 else 99))
    for j in order:
        g = sel_plan[j]
        if g >= 0 and P.O_ok[g]:
            gg, _ = st.candidates(j, np.array([g]))
            if len(gg):
                st.insert(j, g); continue
        rest.append(j)
    left = greedy_repair(st, rest, rng)
    st.refresh_h()
    return st, left


# ------------------------------------------------------------------ P2 ALNS
def alns(P, init_sel, time_limit=10, seed=0, destroys=None, repairs=None, adaptive=True, sub_tl=2.0,
         log_every=1.0):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    ctx = make_ctx(P)
    destroys = destroys or list(DESTROYS); repairs = repairs or list(REPAIRS)
    cur, left = state_from_plan(P, init_sel, rng)
    def f(st):
        return P.evaluate(st.sel, st.hsel)["obj"]
    fc = f(cur); best = cur.copy(); fb = fc
    wd = np.ones(len(destroys)); wr = np.ones(len(repairs))
    sd = np.zeros_like(wd); sr = np.zeros_like(wr); ud = np.zeros_like(wd); ur = np.zeros_like(wr)
    rho, seg = 0.1, 20
    T0 = 0.005 * fc; Tend = 1e-4 * T0
    hist = []; whist = []
    it = 0
    kmin = max(3, int(0.05 * P.nJ)); kmax = max(kmin + 1, min(int(0.2 * P.nJ), 50))
    while time.time() - t0 < time_limit:
        frac = (time.time() - t0) / time_limit
        temp = T0 * (Tend / T0) ** frac
        di = rng.choice(len(destroys), p=wd / wd.sum()) if adaptive else int(rng.integers(len(destroys)))
        ri = rng.choice(len(repairs), p=wr / wr.sum()) if adaptive else int(rng.integers(len(repairs)))
        k = int(rng.integers(kmin, kmax + 1))
        cand = cur.copy()
        rem = DESTROYS[destroys[di]](cand, k, rng, ctx)
        for j in rem:
            cand.remove(j)
        rem = rem + [j for j in range(P.nJ) if cand.sel[j] < 0 and j not in rem]
        if repairs[ri] == "cpsat":
            cpsat_repair(cand, rem, rng, tl=sub_tl)
        else:
            REPAIRS[repairs[ri]](cand, rem, rng)
        cand.refresh_h()
        fcand = f(cand)
        score = 0
        if fcand < fb - 1e-6:
            best = cand.copy(); fb = fcand; cur = cand; fc = fcand; score = 33
        elif fcand < fc - 1e-6:
            cur = cand; fc = fcand; score = 9
        elif rng.random() < math.exp(-(fcand - fc) / max(temp, 1e-9)):
            cur = cand; fc = fcand; score = 13
        sd[di] += score; sr[ri] += score; ud[di] += 1; ur[ri] += 1
        it += 1
        if it % seg == 0 and adaptive:
            m = ud > 0; wd[m] = (1 - rho) * wd[m] + rho * sd[m] / ud[m]
            m = ur > 0; wr[m] = (1 - rho) * wr[m] + rho * sr[m] / ur[m]
            wd = np.maximum(wd, 0.05); wr = np.maximum(wr, 0.05)
            sd[:] = 0; sr[:] = 0; ud[:] = 0; ur[:] = 0
            whist.append(dict(it=it, time=time.time() - t0, **{"d_" + d: w for d, w in zip(destroys, wd / wd.sum())},
                              **{"r_" + r: w for r, w in zip(repairs, wr / wr.sum())}))
        hist.append((it, time.time() - t0, fc, fb))
    return best, dict(iters=it, hist=hist, whist=whist)


# ------------------------------------------------------------------ penalty fitness for GA / SA / NSGA-II
def pfit(P, sel):
    e = P.evaluate(sel)
    return e["obj"] + LAM * e["viol"], e


def genes_to_sel(P, genes):
    return np.array([P.allowed[j][genes[j]] for j in range(P.nJ)])


def sel_to_genes(P, sel):
    out = np.zeros(P.nJ, int)
    for j, g in enumerate(sel):
        w = np.nonzero(P.allowed[j] == g)[0]
        out[j] = w[0] if len(w) else 0
    return out


def ga(P, init_sel, time_limit=10, seed=0, pop=40, p_repair=0.2):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    n = np.array([len(a) for a in P.allowed])
    popg = [sel_to_genes(P, init_sel)]
    while len(popg) < pop:
        st = State(P); greedy_repair(st, list(range(P.nJ)), rng); st.refresh_h()
        if rng.random() < 0.5:
            popg.append(sel_to_genes(P, st.sel))
        else:
            popg.append(rng.integers(0, n))
    popg = np.array(popg)
    fit = np.array([pfit(P, genes_to_sel(P, g))[0] for g in popg])
    hist = []; gen = 0
    while time.time() - t0 < time_limit:
        kids = []
        for _ in range(pop - 2):
            a, b = rng.integers(pop, size=2); p1 = popg[a] if fit[a] < fit[b] else popg[b]
            a, b = rng.integers(pop, size=2); p2 = popg[a] if fit[a] < fit[b] else popg[b]
            mask = rng.random(P.nJ) < 0.5
            c = np.where(mask, p1, p2) if rng.random() < 0.9 else p1.copy()
            mut = rng.random(P.nJ) < 2.0 / P.nJ
            c[mut] = rng.integers(0, n[mut])
            if rng.random() < p_repair:  # repair: keep feasible part, greedily reinsert the rest
                st, _ = state_from_plan(P, genes_to_sel(P, c), rng)
                c = sel_to_genes(P, np.where(st.sel >= 0, st.sel, genes_to_sel(P, c)))
            kids.append(c)
        kids = np.array(kids)
        kf = np.array([pfit(P, genes_to_sel(P, g))[0] for g in kids])
        el = np.argsort(fit)[:2]
        popg = np.vstack([popg[el], kids]); fit = np.concatenate([fit[el], kf])
        gen += 1
        hist.append((gen, time.time() - t0, fit.min()))
    b = int(np.argmin(fit))
    return genes_to_sel(P, popg[b]), dict(iters=gen, hist=hist)


def sa(P, init_sel, time_limit=10, seed=0):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    st, _ = state_from_plan(P, init_sel, rng)
    cur = np.where(st.sel >= 0, st.sel, init_sel)
    cur = np.array([c if c >= 0 else P.allowed[j][0] for j, c in enumerate(cur)])
    fc, _ = pfit(P, cur); best = cur.copy(); fb = fc
    T0 = 0.005 * fc; Tend = 1e-4 * T0
    it = 0; hist = []
    while time.time() - t0 < time_limit:
        frac = (time.time() - t0) / time_limit
        temp = T0 * (Tend / T0) ** frac
        j = int(rng.integers(P.nJ)); al = P.allowed[j]
        new = cur.copy()
        if rng.random() < 0.5:  # shift window by +-1 keeping method
            g = cur[j]; tt = P.O_t[g] + (1 if rng.random() < 0.5 else -1)
            cands = al[(P.O_m[al] == P.O_m[g]) & (P.O_t[al] == tt)]
            if len(cands) == 0:
                continue
            new[j] = cands[0]
        else:
            new[j] = al[int(rng.integers(len(al)))]
        fn, _ = pfit(P, new)
        if fn < fc or rng.random() < math.exp(-(fn - fc) / max(temp, 1e-9)):
            cur, fc = new, fn
            if fc < fb:
                best, fb = cur.copy(), fc
        it += 1
        if it % 50 == 0:
            hist.append((it, time.time() - t0, fb))
    return best, dict(iters=it, hist=hist)
