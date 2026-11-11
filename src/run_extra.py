"""Extra experiments: sensitivity, ablation, Pareto (bi-objective), hard coupling.
usage: python3 run_extra.py sens|ablation|pareto|hard"""
import sys, os, json, time, glob
import numpy as np
from model import *
from heur import *
from sim import simulate
from run_main import ROOT, INSTANCES
import pandas as pd


def record(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)

INST = lambda n: load(os.path.join(INSTANCES, n + ".json"))


def sim_row(inst, P, sel, hsel, mc=3000):
    e = P.evaluate(sel, hsel)
    s = simulate(P, sel, e["hsel"], n=mc, seed=777)
    r = dict(plan_cost=e["nominal"], expected_cost=e["expected"], cc_feasible=int(e["feasible"]), makespan=e["makespan"])
    r.update({"sim_" + k: v for k, v in s.items()})
    return r


def sens():
    out = os.path.join(ROOT, "raw", "sensitivity.csv")
    settings = [("eps", e / 0.05, dict(eps_mult=e / 0.05)) for e in (0.01, 0.05, 0.10, 0.20)]
    settings += [("sla_tight", v, dict(sla_mult=v)) for v in (0.75, 1.0, 1.25)]
    settings += [("bandwidth", v, dict(bw_mult=v)) for v in (0.5, 1.0, 2.0)]
    settings += [("capacity", v, dict(cap_mult=v)) for v in (0.7, 1.0, 1.5)]
    settings += [("deadline_shift", v, dict(Td_shift=v)) for v in (-6, -3, 0)]
    names = [f"S{k:02d}" for k in range(10)] + [f"M{k:02d}" for k in range(3)]
    for n in names:
        inst = INST(n); tl = 15 if n[0] == "S" else 30
        for fac, val, kw in settings:
            if fac != "eps" and n[0] == "M":
                continue
            Pc = Problem(inst, "cc", **kw); Pd = Problem(inst, "det", **kw)
            rows = []
            for name, P in [("P1-SLA-CO-MILP", Pc), ("B2-Det-MILP", Pd)]:
                if name == "B2-Det-MILP" and fac == "eps":
                    if val != 1.0:
                        # B2 does not depend on eps; its plan is the same, evaluated under the eps-specific model
                        pass
                r = solve_cpsat(P, tl, workers=1)
                base = dict(instance=n, factor=fac, value=(val * 0.05 if fac == "eps" else val), method=name,
                            status=r["status"], time=r["time"], gap=r["gap"])
                if r["sel"] is None:
                    rows.append(dict(base, plan_cost=np.nan, expected_cost=np.nan, cc_feasible=0, makespan=np.nan))
                else:
                    rows.append(dict(base, **sim_row(inst, Pc, r["sel"], None)))
            record(rows, out)
        print("sens", n, flush=True)


def ablation():
    out = os.path.join(ROOT, "raw", "ablation.csv")
    variants = {
        "full": dict(),
        "no_cpsat_repair": dict(repairs=["greedy", "regret"]),
        "no_cluster_destroy": dict(destroys=["random", "worst", "window", "product"]),
        "no_adaptivity": dict(adaptive=False),
        "no_chance_constraint": dict(),
    }
    for n in [f"M{k:02d}" for k in range(5)]:
        inst = INST(n); Pc = Problem(inst, "cc"); Pd = Problem(inst, "det")
        st, _ = industry(Pd); b1 = st.sel
        for seed in (0, 1):
            rows = []
            for v, kw in variants.items():
                P = Pd if v == "no_chance_constraint" else Pc
                t0 = time.time(); best, info = alns(P, b1, 8, seed, sub_tl=1.6, **kw)
                rows.append(dict(instance=n, variant=v, seed=seed, time=time.time() - t0, iters=info["iters"],
                                 **sim_row(inst, Pc, best.sel, best.hsel if P is Pc else None, mc=5000)))
            record(rows, out)
        print("abl", n, flush=True)


def nondom(pts):
    pts = sorted(set(pts))
    out = []; best = np.inf
    for mk, c in pts:
        if c < best - 1e-6:
            out.append((mk, c)); best = c
    return out


def pareto():
    from pymoo.core.problem import Problem as PP
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    out = os.path.join(ROOT, "raw", "pareto.csv"); outp = os.path.join(ROOT, "raw", "pareto_points.csv")
    for n in [f"S{k:02d}" for k in range(6)]:
        inst = INST(n); P = Problem(inst, "cc")
        # exact epsilon-constraint on makespan
        exact = []; t0 = time.time(); allopt = True
        r = solve_cpsat(P, 15, workers=1)
        mk_free = P.evaluate(r["sel"])["makespan"]
        for tmax in range(mk_free, 0, -1):
            Pt = Problem(inst, "cc", tmax=tmax)
            rr = solve_cpsat(Pt, 15, workers=1)
            if rr["sel"] is None:
                break
            allopt &= rr["status"] == "OPTIMAL"
            e = P.evaluate(P.plan_from_mt(Pt.mt_of(rr["sel"]))); exact.append((e["makespan"], e["expected"]))
        t_exact = time.time() - t0
        exact = nondom(exact)
        # NSGA-II
        nvar = P.nJ; nb = np.array([len(a) for a in P.allowed])
        class MO(PP):
            def __init__(s):
                super().__init__(n_var=nvar, n_obj=2, n_ieq_constr=1, xl=np.zeros(nvar), xu=nb - 1, vtype=int)
            def _evaluate(s, X, outd, *a, **k):
                F = []; G = []
                for x in X:
                    e = P.evaluate(genes_to_sel(P, x.astype(int)))
                    F.append([e["expected"], e["makespan"]]); G.append([e["viol"]])
                outd["F"] = np.array(F); outd["G"] = np.array(G)
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            init = []
            for k in range(40):  # greedy constructions with random makespan caps + random genes
                if k < 20:
                    cap = int(rng.integers(max(1, exact[0][0] if exact else 3), P.T + 1))
                    Pt = Problem(inst, "cc", tmax=cap)
                    st = State(Pt); greedy_repair(st, list(range(P.nJ)), rng); st.refresh_h()
                    sel = np.array([P.key[(j, Pt.O_m[g], Pt.O_t[g])] if g >= 0 else P.allowed[j][0] for j, g in enumerate(st.sel)])
                    init.append(sel_to_genes(P, sel))
                else:
                    init.append(rng.integers(0, nb))
            alg = NSGA2(pop_size=40, sampling=np.array(init), crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
            t0 = time.time()
            res = minimize(MO(), alg, ("time", 20), seed=seed, verbose=False)
            t_ns = time.time() - t0
            pts = []
            if res.X is not None:
                X = np.atleast_2d(res.X)
                for x in X:
                    e = P.evaluate(genes_to_sel(P, x.astype(int)))
                    if e["feasible"]:
                        pts.append((e["makespan"], e["expected"]))
            ns = nondom(pts)
            allp = np.array(exact + ns, float)
            lo, hi = allp.min(0), allp.max(0); rngv = np.where(hi > lo, hi - lo, 1)
            norm_ = lambda A: (np.array(A, float) - lo) / rngv
            hv = HV(ref_point=np.array([1.1, 1.1]))
            hv_e = hv(norm_(exact)); hv_n = hv(norm_(ns)) if ns else 0.0
            igd = IGD(norm_(exact))(norm_(ns)) if ns else np.nan
            record([dict(instance=n, seed=seed, n_exact=len(exact), exact_all_optimal=int(allopt), t_exact=t_exact, n_nsga=len(ns),
                         t_nsga=t_ns, hv_exact=hv_e, hv_nsga=hv_n, hv_ratio=hv_n / hv_e if hv_e else np.nan, igd=igd)], out)
            record([dict(instance=n, seed=seed, method="NSGA-II", makespan=a, cost=b) for a, b in ns] or
                   [dict(instance=n, seed=seed, method="NSGA-II", makespan=np.nan, cost=np.nan)], outp)
        record([dict(instance=n, seed=-1, method="Exact-eps", makespan=a, cost=b) for a, b in exact], outp)
        print("pareto", n, flush=True)


def hard():
    out = os.path.join(ROOT, "raw", "hard_coupling.csv")
    for n in [f"S{k:02d}" for k in range(10)] + [f"M{k:02d}" for k in range(3)]:
        inst = INST(n)
        rows = []
        for var, hard_ in [("soft", False), ("hard", True)]:
            P = Problem(inst, "cc", hard=hard_)
            r = solve_cpsat(P, 15 if n[0] == "S" else 30, workers=1)
            base = dict(instance=n, variant=var, status=r["status"], time=r["time"], gap=r["gap"])
            if r["sel"] is None:
                rows.append(dict(base, plan_cost=np.nan)); continue
            e = P.evaluate(r["sel"])
            s = simulate(P, r["sel"], e["hsel"], n=3000, seed=777)
            sep = int(sum(abs(e["tau"][a] - e["tau"][b]) > 0 for a, b, k in P.edges if k == "tight"))
            rows.append(dict(base, plan_cost=e["nominal"], expected_cost=e["expected"], makespan=e["makespan"],
                             tight_edges_separated=sep, n_tight=int(P.E_tight.sum()),
                             **{"sim_" + k: v for k, v in s.items()}))
        record(rows, out)
        print("hard", n, flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        globals()[a]()
