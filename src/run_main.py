"""Main experiment: all methods on all instances of the given families.
usage: python3 run_main.py S M   (reads instances/<fam>NN.json; writes results/raw/main_<fam>.csv, plans/, histories/)"""
import sys, os, json, time, csv
import numpy as np
from model import *
from heur import *
from sim import simulate

from paths import RESULTS as ROOT, INSTANCES
BUDGET = {  # exact time limit (s), metaheuristic time (s), seeds, MC scenarios  (scaled to 2 vCPU / 3 h budget)
    "S": dict(tl=15, mh=4, seeds=[0, 1, 2, 3, 4], mc=5000),
    "M": dict(tl=40, mh=8, seeds=[0, 1, 2], mc=5000),
    "L": dict(tl=60, mh=20, seeds=[0, 1, 2], mc=2000),
    "XL": dict(tl=90, mh=30, seeds=[0, 1], mc=2000),
}
WORKERS = 1


def record(rows, path):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def assess(inst, Pc, method, seed, sel, hsel, t, extra, mc):
    e = Pc.evaluate(sel, hsel)
    s = simulate(Pc, sel, e["hsel"], n=mc, seed=777)
    row = dict(instance=inst["name"], family=inst["family"], n_sys=Pc.nS, n_jobs=Pc.nJ, method=method, seed=seed,
               time=round(t, 3), plan_cost=e["nominal"], expected_cost=e["expected"], cc_feasible=int(e["feasible"]),
               viol=e["viol"], v_cc_jobs=e["v_cc"], v_month_h=e["v_month"], v_cap=e["v_cap"], unsched=e["unsched"],
               makespan=e["makespan"], gap=extra.get("gap", ""), status=extra.get("status", ""), iters=extra.get("iters", ""))
    row.update({"sim_" + k: v for k, v in s.items()})
    return row


def save_plan(inst, method, seed, P, sel, hsel):
    d = os.path.join(ROOT, "plans"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{inst['name']}_{method}_{seed}.json"), "w") as f:
        json.dump(dict(mt=P.mt_of(sel), hsel=[int(h) for h in hsel]), f)


def run_instance(path, fam):
    B = BUDGET[fam]
    inst = load(path)
    Pc = Problem(inst, "cc"); Pd = Problem(inst, "det")
    out = os.path.join(ROOT, "raw", f"main_{fam}.csv")
    rows = []
    # B1 industry
    t0 = time.time(); st, forced = industry(Pd); st.refresh_h(); tb1 = time.time() - t0
    b1 = st.sel.copy()
    rows.append(assess(inst, Pc, "B1-Industry", 0, b1, st.hsel, tb1, dict(status=f"forced={forced}"), B["mc"]))
    save_plan(inst, "B1-Industry", 0, Pc, b1, st.hsel)
    # P1 exact CC, B2 exact deterministic
    for name, P in [("P1-SLA-CO-MILP", Pc), ("B2-Det-MILP", Pd)]:
        r = solve_cpsat(P, B["tl"], workers=WORKERS, seed=0, hint=b1)
        if r["sel"] is None:
            rows.append(dict(rows[0], method=name, time=r["time"], plan_cost=np.nan, expected_cost=np.nan, cc_feasible=0,
                             status=r["status"], gap=np.nan, **{k: np.nan for k in rows[0] if k.startswith("sim_")}))
            continue
        rows.append(assess(inst, Pc, name, 0, r["sel"], r["hsel"] if P is Pc else None, r["time"], r, B["mc"]))
        save_plan(inst, name, 0, Pc, r["sel"], Pc.evaluate(r["sel"])["hsel"])
    record(rows, out); rows = []
    # metaheuristics
    for seed in B["seeds"]:
        t0 = time.time(); best, info = alns(Pc, b1, B["mh"], seed, sub_tl=min(2.0, B["mh"] / 5))
        rows.append(assess(inst, Pc, "P2-SLA-ALNS", seed, best.sel, best.hsel, time.time() - t0, info, B["mc"]))
        save_plan(inst, "P2-SLA-ALNS", seed, Pc, best.sel, best.hsel)
        if seed == 0:
            hd = os.path.join(ROOT, "histories"); os.makedirs(hd, exist_ok=True)
            with open(os.path.join(hd, f"alns_{inst['name']}.json"), "w") as f:
                json.dump(dict(hist=info["hist"], whist=info["whist"]), f, default=float)
        t0 = time.time(); s, info = ga(Pc, b1, B["mh"], seed)
        rows.append(assess(inst, Pc, "B3-GA", seed, s, None, time.time() - t0, info, B["mc"]))
        if seed == 0:
            with open(os.path.join(ROOT, "histories", f"ga_{inst['name']}.json"), "w") as f:
                json.dump(dict(hist=info["hist"]), f, default=float)
        t0 = time.time(); s, info = sa(Pc, b1, B["mh"], seed)
        rows.append(assess(inst, Pc, "B4-SA", seed, s, None, time.time() - t0, info, B["mc"]))
        if seed == 0:
            with open(os.path.join(ROOT, "histories", f"sa_{inst['name']}.json"), "w") as f:
                json.dump(dict(hist=info["hist"]), f, default=float)
        record(rows, out); rows = []
    print(inst["name"], "done", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "raw"), exist_ok=True)
    import glob
    for fam in sys.argv[1:]:
        for p in sorted(glob.glob(os.path.join(INSTANCES, f"{fam}[0-9][0-9].json"))):
            run_instance(p, fam)
