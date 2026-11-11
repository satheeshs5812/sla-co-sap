"""Re-simulate saved plans to decompose realised cost into SLA-penalty and non-penalty parts, and evaluate
(i) penalty-multiplier break-even and (ii) a misspecification stress test (throughput CV x1.5, failure prob x2)."""
import os, glob, json
import numpy as np, pandas as pd
from model import Problem, load
from sim import simulate
from paths import RESULTS as R, INSTANCES
rows = []
for f in sorted(glob.glob(os.path.join(R, "plans", "*_0.json"))):
    b = os.path.basename(f)[:-5]; inst_name, method, seed = b.split("_")
    if inst_name[0] not in "SM" or inst_name.startswith("XL"):
        continue
    inst = load(os.path.join(INSTANCES, inst_name + ".json")); P = Problem(inst, "cc")
    pl = json.load(open(f)); sel = P.plan_from_mt([tuple(x) if x else None for x in pl["mt"]]); h = np.array(pl["hsel"])
    for sc, kw in [("nominal", {}), ("stress_cv1.5_pf2", dict(cv_mult=1.5, pf_mult=2.0))]:
        s = simulate(P, sel, h, n=4000, seed=777, **kw)
        rows.append(dict(instance=inst_name, family=inst_name[0], method=method, scenario=sc, cost_mean=s["cost_mean"], pen_mean=s["pen_mean"],
                         cvar95=s["cvar95"], prd_ev_viol_rate=s["prd_ev_viol_rate"], month_viol_rate=s["month_viol_rate"], ev_viol_rate=s["ev_viol_rate"]))
d = pd.DataFrame(rows); d.to_csv(os.path.join(R, "ALL_postsim_penalty_stress.csv"), index=False)
print(d.groupby(["family", "scenario", "method"])[["cost_mean", "pen_mean", "prd_ev_viol_rate", "month_viol_rate"]].mean())
