"""Smoke test: run every method of run_main.py on one instance with a short budget.

usage: python src/smoke_test.py [INSTANCE] [OUTDIR]     (defaults: S00, a fresh temporary directory)

Uses the unchanged algorithms and seeds of run_main.py; only the time limits, the number of seeds and the
number of Monte Carlo scenarios are reduced, so the numbers are NOT those of the paper. Output goes to
OUTDIR (never to results/).
"""
import os, sys, tempfile, time

name = sys.argv[1] if len(sys.argv) > 1 else "S00"
out = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="slaco_smoke_")
os.environ["SLACO_RESULTS"] = os.path.abspath(out)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import run_main  # noqa: E402

fam = name.rstrip("0123456789")
run_main.BUDGET[fam] = dict(tl=5, mh=2, seeds=[0], mc=500)
os.makedirs(os.path.join(run_main.ROOT, "raw"), exist_ok=True)
t0 = time.time()
run_main.run_instance(os.path.join(run_main.INSTANCES, name + ".json"), fam)
d = pd.read_csv(os.path.join(run_main.ROOT, "raw", f"main_{fam}.csv"))
print(d[["method", "seed", "time", "plan_cost", "cc_feasible", "sim_cost_mean", "sim_prd_ev_viol_rate"]].to_string(index=False))
print(f"smoke test finished in {time.time() - t0:.1f} s; output in {run_main.ROOT}")
