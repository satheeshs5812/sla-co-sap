"""Problem preprocessing (chance-constraint deterministic equivalents), evaluator and CP-SAT model."""
import json, math, time
import numpy as np
from scipy.stats import norm
from ortools.sat.python import cp_model
from calib import *

H = HYPERSCALERS
BIG_UNSCHED = 5e6


def load(path):
    with open(path) as f:
        return json.load(f)


def lognorm_params(mean, cv):
    s2 = math.log(1 + cv * cv)
    return math.log(mean) - s2 / 2, math.sqrt(s2)


def prob_exceed(setup, V, mu, sig, D):
    """P(setup + V/theta > D), theta ~ LN(mu, sig). vectorised in D."""
    D = np.asarray(D, float)
    out = np.ones_like(D)
    pos = D > setup
    out[pos] = norm.cdf((np.log(V / (D[pos] - setup)) - mu) / sig)
    return out


class Problem:
    def __init__(self, inst, mode="cc", eps_mult=1.0, sla_mult=1.0, hard=False, cap_mult=1.0, bw_mult=1.0,
                 Td_shift=0, tmax=None):
        self.inst, self.mode, self.hard = inst, mode, hard
        S = inst["systems"]; W = inst["windows"]; T = inst["T"]
        self.T, self.nS = T, len(S)
        self.Td = inst["Td"] + Td_shift
        self.S = S
        self.L = np.array([0] + [w["L"] for w in W])
        self.month = np.array([0] + [w["month"] for w in W])
        self.nM = int(self.month.max()) + 1
        self.C = np.array([0] + [w["C"] * cap_mult for w in W], float)
        self.Bcap = np.array([0] + [w["B"] * bw_mult * GB_PER_GBPS_HOUR * w["L"] * WAN_EFF for w in W], float)
        self.Pmax = np.array([0] + [w["P"] for w in W])
        blk = np.array([False] + [w["blackout_fin"] for w in W])
        # system-level data
        self.D = np.array([SLA[s["sla"]]["D"] * sla_mult for s in S])
        self.budget = np.array([(1 - SLA[s["sla"]]["A"]) * MONTH_HOURS * sla_mult for s in S])
        self.eps = np.array([min(0.5, SLA[s["sla"]]["eps"] * eps_mult) for s in S])
        self.pen = np.array([SLA[s["sla"]]["pen"] for s in S], float)
        util = np.array([1.0 if s["tier"] == "PRD" else NONPROD_UTIL for s in S])
        price = np.full((self.nS, 3), np.inf)
        for i, s in enumerate(S):
            for hi, h in enumerate(H):
                c = [k[5] for k in SKUS if k[0] == h and k[4] >= s["saps"] and k[3] >= s["mem"]]
                if c:
                    price[i, hi] = min(c) * util[i]
            if s["already_cloud"]:
                price[i, :] = np.inf; price[i, H.index(s["cloud_h"])] = 0.0
        self.price = price
        self.migr = np.array([not s["already_cloud"] for s in S])
        minp = np.where(self.migr, np.min(np.where(np.isfinite(price), price, 1e9), axis=1), 0.0)
        self.onprem_week = ONPREM_FACTOR * 168 * minp
        self.cloud_week = 168 * price  # per system per h
        # co-location components over tight edges
        parent = list(range(self.nS))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        self.edges = [(e["a"], e["b"], e["kind"]) for e in inst["edges"]
                      if not S[e["a"]]["already_cloud"] and not S[e["b"]]["already_cloud"]]
        for a, b, k in self.edges:
            if k == "tight":
                parent[find(a)] = find(b)
        roots = sorted(set(find(i) for i in range(self.nS)))
        rid = {r: c for c, r in enumerate(roots)}
        self.comp = np.array([rid[find(i)] for i in range(self.nS)])
        self.nC = len(roots)
        self.comp_members = [[] for _ in range(self.nC)]
        for i in range(self.nS):
            self.comp_members[self.comp[i]].append(i)
        self.comp_allowed = []
        for c in range(self.nC):
            al = [hi for hi in range(3) if all(np.isfinite(price[i, hi]) for i in self.comp_members[c])]
            if not al:  # should not happen; fall back to the h feasible for most members
                al = [int(np.argmax([sum(np.isfinite(price[i, hi]) for i in self.comp_members[c]) for hi in range(3)]))]
            self.comp_allowed.append(al)
        self.nbrs = [[] for _ in range(self.nS)]
        for a, b, k in self.edges:
            self.nbrs[a].append((b, HYBRID_PEN[k], k)); self.nbrs[b].append((a, HYBRID_PEN[k], k))
        self.E_a = np.array([e[0] for e in self.edges], int); self.E_b = np.array([e[1] for e in self.edges], int)
        self.E_pen = np.array([HYBRID_PEN[e[2]] for e in self.edges]); self.E_tight = np.array([e[2] == "tight" for e in self.edges], bool)
        # precedence (transition jobs) DEV->QAS->PRD
        self.pred = [None] * self.nS; self.succ = [None] * self.nS
        land = {}
        for s in S:
            land.setdefault(s["landscape"], {})[s["tier"]] = s["id"]
        self.lag = np.zeros(self.nS, int)
        for li, d in land.items():
            lag = inst["lags"][str(li)] if str(li) in inst["lags"] else inst["lags"][li]
            for a, b in [("DEV", "QAS"), ("QAS", "PRD")]:
                self.pred[d[b]] = d[a]; self.succ[d[a]] = d[b]; self.lag[d[b]] = lag
        self.P_b = np.array([b for b in range(self.nS) if self.pred[b] is not None], int)
        self.P_a = np.array([self.pred[b] for b in self.P_b], int)
        # jobs
        nQ = (T + 12) // 13
        jobs = []
        for s in S:
            jobs.append(dict(sys=s["id"], kind="T", jtype=s["jtype"], q=-1))
            for q in range(nQ):
                jobs.append(dict(sys=s["id"], kind="M", jtype="MAINT", q=q))
        self.jobs = jobs; self.nJ = len(jobs)
        self.trans_job = np.zeros(self.nS, int)
        for j, jb in enumerate(jobs):
            if jb["kind"] == "T":
                self.trans_job[jb["sys"]] = j
        # options (all structurally feasible (j,m,t))
        rows = []
        mnames = list(METHODS)
        self.mnames = mnames
        tiers = {"DEV": 0.5, "QAS": 0.7, "PRD": 1.0}
        self.qcache = {}
        for j, jb in enumerate(jobs):
            i = jb["sys"]; s = S[i]
            if jb["kind"] == "T":
                last = self.Td if s["product"] == "ERP" else T
                if tmax is not None:
                    last = min(last, tmax)
                ts = list(range(1, last + 1))
            else:
                ts = [t for t in range(1, T + 1) if (t - 1) // 13 == jb["q"]]
            if s["fin"]:
                ts = [t for t in ts if not blk[t]]
            for mi, mn in enumerate(mnames):
                M = METHODS[mn]
                if M["jtype"] != jb["jtype"]:
                    continue
                if M["prereq"] == "HANA" and s["src"] != "HANA" and not (jb["kind"] == "M" and False):
                    continue
                if M["prereq"] == "anyDB" and s["src"] != "anyDB":
                    continue
                if M["prd_only"] and s["tier"] != "PRD":
                    continue
                V = s["db"]; mu, sig = lognorm_params(M["thr"], M["cv"])
                nominal = M["setup_h"] + V / M["thr"]
                Dl = np.minimum(self.D[i], self.L[ts])
                p = M["pfail"]
                ps = prob_exceed(M["setup_h"], V, mu, sig, Dl)
                if M["repl"]:
                    pf = (M["setup_h"] + M["rollback_h"] > Dl).astype(float)
                else:
                    pf = prob_exceed(M["setup_h"] + M["rollback_h"], FAIL_KAPPA * V, mu, sig, Dl)
                pv = (1 - p) * ps + p * pf
                qcc = self.mix_quantile(M, V, mu, sig, 1 - self.eps[i])
                tf = tiers[s["tier"]]
                lab = M["labour"] * tf
                for k, t in enumerate(ts):
                    rows.append((j, mi, t, i, jb["kind"] == "T", pv[k], qcc, nominal, lab,
                                 V * M["transfer"] if jb["kind"] == "T" else 0.0, M["service"], p, M["presync"]))
        R = np.array(rows, dtype=float)
        self.O_job = R[:, 0].astype(int); self.O_m = R[:, 1].astype(int); self.O_t = R[:, 2].astype(int)
        self.O_sys = R[:, 3].astype(int); self.O_trans = R[:, 4].astype(bool)
        self.O_pv = R[:, 5]; self.O_qcc = R[:, 6]; self.O_nom = R[:, 7]; self.O_eff = R[:, 8]
        self.O_trf = R[:, 9]; self.O_service = R[:, 10]; self.O_pf = R[:, 11]; presync = R[:, 12]
        self.O_month = self.month[self.O_t]
        i = self.O_sys; t = self.O_t
        Dl = np.minimum(self.D[i], self.L[t])
        self.O_ok_cc = self.O_pv <= self.eps[i] + 1e-12
        self.O_ok_det = self.O_nom <= Dl
        # nominal (plan) cost: service + labour + on-prem run until cutover (incl. cutover week) + presync dual run
        nomc = self.O_service + self.O_eff * LABOUR_RATE
        tr = self.O_trans & self.migr[i]
        nomc = nomc + np.where(tr, self.onprem_week[i] * t + presync * 168 * np.where(self.migr[i], self.onprem_week[i] / (ONPREM_FACTOR * 168), 0), 0)
        self.O_nomcost = nomc
        self.O_expextra = self.O_pf * (0.5 * self.O_service + self.O_eff * LABOUR_RATE + np.where(tr, self.onprem_week[i], 0)) + self.O_pv * self.pen[i]
        if mode == "cc":
            self.O_ok = self.O_ok_cc; self.O_q = self.O_qcc; self.O_cost = nomc + self.O_expextra
        else:
            self.O_ok = self.O_ok_det; self.O_q = self.O_nom; self.O_cost = nomc
        self.O_qmin = np.ceil(self.O_q * 60).astype(int)
        self.budget_min = np.floor(self.budget * 60).astype(int)
        # per-job slices
        self.off = np.searchsorted(self.O_job, np.arange(self.nJ))
        self.end = np.searchsorted(self.O_job, np.arange(self.nJ), side="right")
        self.allowed = [np.nonzero(self.O_ok[self.off[j]:self.end[j]])[0] + self.off[j] for j in range(self.nJ)]
        self.job_sys = np.array([jb["sys"] for jb in jobs])
        self.job_trans = np.array([jb["kind"] == "T" for jb in jobs])
        self.infeasible_jobs = [j for j in range(self.nJ) if len(self.allowed[j]) == 0]
        self.key = {}
        for g in range(len(self.O_job)):
            self.key[(self.O_job[g], self.O_m[g], self.O_t[g])] = g

    def mix_quantile(self, M, V, mu, sig, level):
        p = M["pfail"]
        def cdf(setup, VV, x):  # P(setup + VV/theta <= x)
            if x <= setup:
                return 0.0
            z = (math.log(VV / (x - setup)) - mu) / sig
            return 1 - 0.5 * math.erfc(-z / math.sqrt(2))
        def F(x):
            ps = cdf(M["setup_h"], V, x)
            if M["repl"]:
                pf = 1.0 if x >= M["setup_h"] + M["rollback_h"] else 0.0
            else:
                pf = cdf(M["setup_h"] + M["rollback_h"], FAIL_KAPPA * V, x)
            return (1 - p) * ps + p * pf
        lo, hi = M["setup_h"], M["setup_h"] + M["rollback_h"] + V / math.exp(mu - 8 * sig) + 1
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if F(mid) >= level:
                hi = mid
            else:
                lo = mid
        return hi

    # ------------------------------------------------------------ evaluation
    def best_h(self, tau):
        """optimal hyperscaler per component given transition windows tau (array nS, 0 = unscheduled)."""
        rem = np.where(tau > 0, self.T - tau, 0)
        cost = np.zeros((self.nC, 3))
        cw = np.where(np.isfinite(self.cloud_week), self.cloud_week, 1e12)
        np.add.at(cost, self.comp, cw * rem[:, None])
        hsel = np.zeros(self.nC, int)
        for c in range(self.nC):
            al = self.comp_allowed[c]
            hsel[c] = al[int(np.argmin(cost[c, al]))]
        return hsel

    def evaluate(self, sel, hsel=None, use_ok=None):
        """sel: array nJ of global option idx (-1 unscheduled). Returns dict with cost (this problem's objective),
        nominal plan cost, expected cost (cc objective terms), violation measures."""
        sel = np.asarray(sel)
        sc = sel >= 0
        g = sel[sc]
        tau = np.zeros(self.nS, int)
        trg = g[self.O_trans[g]]
        tau[self.O_sys[trg]] = self.O_t[trg]
        if hsel is None:
            hsel = self.best_h(tau)
        hs = hsel[self.comp]
        rem = np.where(tau > 0, self.T - tau, 0)
        cloud = np.where(self.migr, self.cloud_week[np.arange(self.nS), hs] * rem, 0.0)
        cloud = np.where(np.isfinite(cloud), cloud, 1e9).sum()
        ea, eb = self.E_a, self.E_b
        both = (tau[ea] > 0) & (tau[eb] > 0)
        dd = np.abs(tau[ea] - tau[eb]) * both
        if self.hard:
            hard_v = int(((dd > 0) & self.E_tight).sum()); hyb = float((dd * self.E_pen * ~self.E_tight).sum())
        else:
            hard_v = 0; hyb = float((dd * self.E_pen).sum())
        nom = self.O_nomcost[g].sum() + cloud + hyb
        expc = nom + self.O_expextra[g].sum()
        obj = (expc if self.mode == "cc" else nom) + BIG_UNSCHED * (~sc).sum()
        # violations
        T1 = self.T + 1
        eff = np.bincount(self.O_t[g], self.O_eff[g], T1); trf = np.bincount(self.O_t[g], self.O_trf[g], T1)
        par = np.bincount(self.O_t[g], self.O_trans[g].astype(float), T1)
        v_cap = np.maximum(eff - self.C - 1e-6, 0).sum() / max(self.C[1], 1) + np.maximum(trf - self.Bcap - 1e-6, 0).sum() / max(self.Bcap[1], 1) \
            + np.maximum(par - self.Pmax, 0).sum()
        busy = np.bincount(self.O_sys[g] * T1 + self.O_t[g], minlength=self.nS * T1)
        v_busy = np.maximum(busy - 1, 0).sum()
        md = np.bincount(self.O_sys[g] * self.nM + self.O_month[g], self.O_qmin[g], self.nS * self.nM).reshape(self.nS, self.nM)
        v_month = np.maximum(md - self.budget_min[:, None], 0).sum() / 60.0
        pa, pb = self.P_a, self.P_b
        okp = (tau[pa] > 0) & (tau[pb] > 0)
        v_prec = int(np.maximum(tau[pa] + self.lag[pb] - tau[pb], 0)[okp].sum())
        v_cc = int((~self.O_ok[g]).sum())
        hok = bool(np.isfinite(self.price[np.arange(self.nS), hs][self.migr]).all())
        n_un = int((~sc).sum())
        viol = v_cap + v_busy + v_month + v_prec + v_cc + hard_v + n_un + (0 if hok else 1)
        mk = int(tau.max())
        return dict(obj=obj, nominal=nom, expected=expc, viol=viol, v_cap=v_cap, v_busy=int(v_busy), v_month=v_month,
                    v_prec=v_prec, v_cc=v_cc, v_hard=hard_v, unsched=n_un, makespan=mk, hsel=hsel, tau=tau,
                    feasible=viol < 1e-9)

    def plan_from_mt(self, mt):
        """mt: list of (m_idx, t) per job (or None) -> sel array in this problem's option table (-1 if structurally impossible)."""
        sel = np.full(self.nJ, -1)
        for j, v in enumerate(mt):
            if v is not None:
                sel[j] = self.key.get((j, v[0], v[1]), -1)
        return sel

    def mt_of(self, sel):
        return [(int(self.O_m[g]), int(self.O_t[g])) if g >= 0 else None for g in sel]


# ---------------------------------------------------------------- CP-SAT
def solve_cpsat(P, time_limit=60, workers=2, seed=0, allowed=None, fixed_h=None, hint=None, log=False):
    """Exact SLA-CO model. allowed: per-job arrays of global option indices (default P.allowed).
    fixed_h: dict comp->h for components whose hyperscaler is fixed."""
    t0 = time.time()
    allowed = allowed if allowed is not None else P.allowed
    for j in range(P.nJ):
        if len(allowed[j]) == 0:
            return dict(status="INFEASIBLE_PRE", sel=None, obj=None, bound=None, gap=None, time=time.time() - t0)
    m = cp_model.CpModel()
    x = {}
    T1 = P.T + 1
    eff = [[] for _ in range(T1)]; trf = [[] for _ in range(T1)]; par = [[] for _ in range(T1)]
    busy = {}; mon = {}
    obj = []
    tau = [None] * P.nS
    xt = {}  # (sys, t) -> list of trans vars
    for j in range(P.nJ):
        vs = []
        for g in allowed[j]:
            v = m.NewBoolVar(""); x[g] = v; vs.append(v)
            t = P.O_t[g]; i = P.O_sys[g]
            if P.O_eff[g] > 0: eff[t].append((int(math.ceil(P.O_eff[g] * 10 - 1e-9)), v))
            if P.O_trf[g] > 0: trf[t].append((int(math.ceil(P.O_trf[g])), v))
            if P.O_trans[g]:
                par[t].append(v); xt.setdefault((i, t), []).append(v)
            busy.setdefault((i, t), []).append(v)
            if P.O_qmin[g] > 0: mon.setdefault((i, P.O_month[g]), []).append((int(P.O_qmin[g]), v))
            obj.append((int(round(P.O_cost[g])), v))
        m.AddExactlyOne(vs)
        if P.job_trans[j]:
            tau[P.job_sys[j]] = sum(int(P.O_t[g]) * x[g] for g in allowed[j])
    for t in range(1, T1):
        if eff[t] and sum(c for c, _ in eff[t]) > P.C[t] * 10:
            m.Add(sum(c * v for c, v in eff[t]) <= int(P.C[t] * 10))
        if trf[t] and sum(c for c, _ in trf[t]) > P.Bcap[t]:
            m.Add(sum(c * v for c, v in trf[t]) <= int(P.Bcap[t]))
        if len(par[t]) > P.Pmax[t]:
            m.Add(sum(par[t]) <= int(P.Pmax[t]))
    for k, vs in busy.items():
        if len(vs) > 1: m.AddAtMostOne(vs)
    for (i, mo), lst in mon.items():
        if sum(c for c, _ in lst) > P.budget_min[i]:
            m.Add(sum(c * v for c, v in lst) <= int(P.budget_min[i]))
    for b in range(P.nS):
        a = P.pred[b]
        if a is not None:
            m.Add(tau[b] >= tau[a] + int(P.lag[b]))
    for a, b, k in P.edges:
        if P.hard and k == "tight":
            m.Add(tau[a] == tau[b])
        else:
            d = m.NewIntVar(0, P.T, "")
            m.Add(d >= tau[a] - tau[b]); m.Add(d >= tau[b] - tau[a])
            obj.append((int(HYBRID_PEN[k]), d))
    # hyperscaler selection per component + cloud run cost linearisation
    y = {}
    for c in range(P.nC):
        al = P.comp_allowed[c]
        if fixed_h is not None and c in fixed_h:
            al = [fixed_h[c]]
        for h in al:
            y[c, h] = m.NewBoolVar("")
        m.AddExactlyOne([y[c, h] for h in al])
    for i in range(P.nS):
        if not P.migr[i]:
            continue
        c = P.comp[i]; al = [h for h in range(3) if (c, h) in y]
        j = P.trans_job[i]
        ts = sorted(set(int(P.O_t[g]) for g in allowed[j]))
        if len(al) == 1:
            h = al[0]
            for t in ts:
                for v in xt[(i, t)]:
                    obj.append((int(round(P.cloud_week[i, h] * (P.T - t))), v))
            continue
        w = {}
        for h in al:
            for t in ts:
                w[h, t] = m.NewBoolVar("")
                obj.append((int(round(P.cloud_week[i, h] * (P.T - t))), w[h, t]))
            m.Add(sum(w[h, t] for t in ts) == y[c, h])
        for t in ts:
            m.Add(sum(w[h, t] for h in al) == sum(xt[(i, t)]))
    m.Minimize(sum(c * v for c, v in obj))
    if hint is not None:
        for j, g in enumerate(hint):
            if g in x:
                for g2 in allowed[j]:
                    m.AddHint(x[g2], 1 if g2 == g else 0)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(time_limit)
    sv.parameters.num_workers = workers
    sv.parameters.random_seed = seed
    sv.parameters.log_search_progress = log
    st = sv.Solve(m)
    name = sv.StatusName(st)
    out = dict(status=name, time=time.time() - t0, sel=None, obj=None, bound=None, gap=None)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = np.full(P.nJ, -1)
        for j in range(P.nJ):
            for g in allowed[j]:
                if sv.Value(x[g]):
                    sel[j] = g; break
        hsel = np.zeros(P.nC, int)
        for (c, h), v in y.items():
            if sv.Value(v):
                hsel[c] = h
        o, b = sv.ObjectiveValue(), sv.BestObjectiveBound()
        out.update(sel=sel, hsel=hsel, obj=o, bound=b, gap=(o - b) / max(abs(o), 1e-9))
    return out
