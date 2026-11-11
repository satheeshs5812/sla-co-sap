"""Aggregate raw CSVs -> ALL_*.csv, tables (markdown fragments), statistics and figures."""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS as R, INSTANCES
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)
METH = ["P1-SLA-CO-MILP", "P2-SLA-ALNS", "B1-Industry", "B2-Det-MILP", "B3-GA", "B4-SA"]
COL = {"P1-SLA-CO-MILP": "#0072B2", "P2-SLA-ALNS": "#D55E00", "B1-Industry": "#000000", "B2-Det-MILP": "#56B4E9",
       "B3-GA": "#009E73", "B4-SA": "#CC79A7"}
MK = {"P1-SLA-CO-MILP": "s", "P2-SLA-ALNS": "D", "B1-Industry": "X", "B2-Det-MILP": "o", "B3-GA": "^", "B4-SA": "v"}
FAMS = ["S", "M", "L", "XL"]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
                     "figure.dpi": 100, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
OUT = []  # markdown fragments


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300); fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def md(df, floatfmt=None):
    return df.to_markdown(index=False, floatfmt=floatfmt or ".3g")


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = r.choice(x, size=(B, len(x)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main_tables():
    fs = sorted(glob.glob(os.path.join(R, "raw", "main_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df.to_csv(os.path.join(R, "ALL_main_runs.csv"), index=False)
    num = [c for c in df.columns if c not in ("instance", "family", "method", "status", "gap", "iters")]
    df["gap"] = pd.to_numeric(df["gap"], errors="coerce")
    # per instance x method (mean over seeds)
    g = df.groupby(["family", "instance", "method"])
    inst = g[num + ["gap"]].mean().reset_index()
    inst["status"] = g["status"].first().values
    # relative deviation (%) from the best value per instance among methods for cost metrics
    for c in ["plan_cost", "expected_cost", "sim_cost_mean", "sim_cvar95"]:
        best = inst.groupby("instance")[c].transform("min")
        inst["rpd_" + c] = 100 * (inst[c] - best) / best
    inst.to_csv(os.path.join(R, "ALL_main_per_instance.csv"), index=False)
    rows = []
    for fam in FAMS:
        for m in METH:
            d = inst[(inst.family == fam) & (inst.method == m)]
            if d.empty:
                continue
            ci = boot_ci(d["rpd_sim_cost_mean"])
            rows.append(dict(family=fam, method=m, n=len(d),
                             plan_cost_M=f"{d.plan_cost.mean()/1e6:.3f} ± {d.plan_cost.std()/1e6:.3f}",
                             exp_cost_M=f"{d.expected_cost.mean()/1e6:.3f}",
                             real_cost_M=f"{d.sim_cost_mean.mean()/1e6:.3f} ± {d.sim_cost_mean.std()/1e6:.3f}",
                             cvar95_M=f"{d.sim_cvar95.mean()/1e6:.3f}",
                             rpd_real_pct=f"{d.rpd_sim_cost_mean.mean():.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                             rpd_cvar_pct=f"{d.rpd_sim_cvar95.mean():.2f}",
                             ev_viol_pct=f"{100*d.sim_ev_viol_rate.mean():.2f}",
                             prd_ev_viol_pct=f"{100*d.sim_prd_ev_viol_rate.mean():.2f} ± {100*d.sim_prd_ev_viol_rate.std():.2f}",
                             month_viol_pct=f"{100*d.sim_month_viol_rate.mean():.2f}",
                             cc_feas_pct=f"{100*d.cc_feasible.mean():.0f}",
                             makespan=f"{d.makespan.mean():.1f}",
                             real_makespan=f"{d.sim_makespan_mean.mean():.1f}",
                             time_s=f"{d.time.mean():.1f}",
                             gap_pct=(f"{100*d.gap.mean():.2f}" if d.gap.notna().any() else "-")))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(R, "ALL_main_summary.csv"), index=False)
    OUT.append("### Main results (mean ± sd over instances; seeded methods first averaged over seeds)\n\n"
               "Costs in M USD. rpd = relative % deviation from the best method on the same instance (95% bootstrap CI of the mean). "
               "ev_viol = realised per-event SLA violation rate (all tiers); prd_ev_viol = PRD events only; month_viol = system-months "
               "breaching the planned-downtime availability budget; cc_feas = plan satisfies all chance constraints and hard constraints of the SLA-CO model.\n\n" + md(T) + "\n")
    # P1 status
    st = inst[inst.method == "P1-SLA-CO-MILP"][["family", "instance", "n_sys", "status", "gap", "time"]]
    OUT.append("### P1 solver status per instance\n\n" + md(st) + "\n")
    st2 = inst[inst.method == "B2-Det-MILP"][["family", "instance", "status", "gap", "time"]]
    OUT.append("### B2 solver status per instance\n\n" + md(st2) + "\n")
    return df, inst


def statistics(inst):
    res = []
    OUT.append("### Statistical tests\n")
    for metric in ["sim_cost_mean", "sim_cvar95", "sim_prd_ev_viol_rate"]:
        for fam in FAMS + ["ALL"]:
            d = inst if fam == "ALL" else inst[inst.family == fam]
            piv = d.pivot(index="instance", columns="method", values=metric)[[m for m in METH if m in d.method.unique()]].dropna()
            if len(piv) < 3 or piv.shape[1] < 3:
                continue
            fr = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
            ranks = piv.rank(axis=1).mean()
            res.append(dict(metric=metric, family=fam, N=len(piv), friedman_chi2=fr.statistic, p=fr.pvalue,
                            **{"rank_" + c.split("-")[0]: r for c, r in ranks.items()}))
            if fam == "ALL" and metric == "sim_cost_mean":
                nem = sp.posthoc_nemenyi_friedman(piv.values)
                nem.index = nem.columns = piv.columns
                nem.to_csv(os.path.join(R, "ALL_nemenyi_realised_cost.csv"))
                cd_diagram(ranks, len(piv), "fig9_cd_diagram", "Realised mean cost (all families)")
            if fam == "ALL" and metric == "sim_prd_ev_viol_rate":
                cd_diagram(ranks, len(piv), "fig9b_cd_diagram_violation", "Realised PRD violation rate (all families)")
    F = pd.DataFrame(res)
    F.to_csv(os.path.join(R, "ALL_friedman.csv"), index=False)
    OUT.append("Friedman tests (block = instance; mean ranks, lower = better):\n\n" + md(F, ".4g") + "\n")
    # Wilcoxon P2 vs each baseline, Holm
    W = []
    for metric in ["sim_cost_mean", "sim_prd_ev_viol_rate", "sim_cvar95"]:
        piv = inst.pivot(index="instance", columns="method", values=metric)
        ps = []
        for b in [m for m in METH if m != "P2-SLA-ALNS"]:
            x = piv[["P2-SLA-ALNS", b]].dropna()
            diff = x["P2-SLA-ALNS"] - x[b]
            if (diff != 0).sum() < 1:
                p = 1.0; stat = np.nan
            else:
                t = stats.wilcoxon(x["P2-SLA-ALNS"], x[b], zero_method="wilcox")
                p, stat = t.pvalue, t.statistic
            W.append(dict(metric=metric, comparison=f"P2 vs {b}", N=len(x), median_diff=float(np.median(diff)),
                          P2_better=int((diff < 0).sum()), ties=int((diff == 0).sum()), P2_worse=int((diff > 0).sum()), W=stat, p=p))
    W = pd.DataFrame(W)
    W["p_holm"] = np.nan
    for metric in W.metric.unique():
        idx = W.index[W.metric == metric]
        p = W.loc[idx, "p"].values; o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0
        for r, k in enumerate(o):
            run = max(run, min(1, (m - r) * p[k])); adj[k] = run
        W.loc[idx, "p_holm"] = adj
    W.to_csv(os.path.join(R, "ALL_wilcoxon_holm.csv"), index=False)
    OUT.append("Wilcoxon signed-rank, P2 vs each method, pooled over all instances of all families (paired by instance), Holm-adjusted per metric:\n\n" + md(W, ".4g") + "\n")


def cd_diagram(ranks, N, name, title):
    k = len(ranks)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}[k]
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    r = ranks.sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.2, hi + 0.2); ax.set_ylim(0, 1)
    ax.hlines(0.8, lo, hi, color="k", lw=1)
    for v in range(lo, hi + 1):
        ax.vlines(v, 0.78, 0.82, color="k"); ax.text(v, 0.86, str(v), ha="center")
    ax.hlines(0.95, lo, lo + cd, color="k", lw=2); ax.text(lo + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for n_, (m, v) in enumerate(r.items()):
        left = n_ < k / 2
        y = 0.6 - 0.12 * (n_ if left else k - 1 - n_)
        xt = lo - 0.1 if left else hi + 0.1
        ax.plot([v, v], [0.8, y], color=COL.get(m, "k"), lw=1)
        ax.plot([v, xt], [y, y], color=COL.get(m, "k"), lw=1)
        ax.text(xt, y, f"{m} ({v:.2f})", ha="right" if left else "left", va="center", fontsize=8, color=COL.get(m, "k"))
    # cliques: groups whose rank difference < CD
    vals = r.values; ys = 0.72
    for a in range(k):
        b = max([bb for bb in range(a, k) if vals[bb] - vals[a] < cd])
        if b > a and not any(a2 <= a and b2 >= b and (a2, b2) != (a, b) for a2, b2 in getattr(cd_diagram, "_seen", [])):
            ax.hlines(ys, vals[a] - 0.03, vals[b] + 0.03, color="gray", lw=3); ys -= 0.04
    ax.axis("off"); ax.set_title(f"{title}; Nemenyi CD (α=0.05), N={N}")
    fig.subplots_adjust(left=0.25, right=0.75)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(inst):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for m in METH:
        d = inst[inst.method == m]
        axs[0].scatter(d.rpd_plan_cost, 100 * d.sim_prd_ev_viol_rate, s=14, alpha=0.35, color=COL[m], marker=MK[m])
        for fam in FAMS:
            e = d[d.family == fam]
            if len(e):
                axs[0].scatter(e.rpd_plan_cost.mean(), 100 * e.sim_prd_ev_viol_rate.mean(), s=70, color=COL[m], marker=MK[m],
                               edgecolor="k", label=m if fam == "S" else None)
                axs[0].annotate(fam, (e.rpd_plan_cost.mean(), 100 * e.sim_prd_ev_viol_rate.mean()), fontsize=7, xytext=(4, 3), textcoords="offset points")
        axs[1].scatter(d.rpd_plan_cost, d.rpd_sim_cost_mean, s=14, alpha=0.5, color=COL[m], marker=MK[m], label=m)
    axs[0].set_xlabel("Plan (nominal) cost, % above best plan on instance"); axs[0].set_ylabel("Realised PRD event SLA-violation rate (%)")
    axs[0].set_xscale("symlog", linthresh=1); axs[0].legend(loc="upper right"); axs[0].set_title("(a) plan cost vs realised violations (large = family mean)")
    axs[1].set_xlabel("Plan (nominal) cost, % above best"); axs[1].set_ylabel("Realised mean cost (MC), % above best")
    axs[1].set_xscale("symlog", linthresh=1); axs[1].set_yscale("symlog", linthresh=1); axs[1].plot([0, 100], [0, 100], "k:", lw=0.8)
    axs[1].set_title("(b) plan cost vs realised cost (per instance)")
    save(fig, "fig2_cost_vs_violation")


def fig_costdist(inst):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.8), sharey=False)
    for a, fam in zip(axs, FAMS):
        d = inst[inst.family == fam]
        ms = [m for m in METH if m in d.method.unique()]
        if not ms:
            a.axis("off"); continue
        data = [d[d.method == m].rpd_sim_cost_mean.values for m in ms]
        cv = [d[d.method == m].rpd_sim_cvar95.values for m in ms]
        bp = a.boxplot(data, positions=np.arange(len(ms)) - 0.18, widths=0.32, patch_artist=True, showfliers=False)
        for p_, m in zip(bp["boxes"], ms):
            p_.set_facecolor(COL[m]); p_.set_alpha(0.6)
        bp2 = a.boxplot(cv, positions=np.arange(len(ms)) + 0.18, widths=0.32, patch_artist=True, showfliers=False)
        for p_, m in zip(bp2["boxes"], ms):
            p_.set_facecolor("white"); p_.set_edgecolor(COL[m]); p_.set_hatch("///")
        a.set_xticks(range(len(ms))); a.set_xticklabels([m.split("-")[0] for m in ms])
        a.set_yscale("symlog", linthresh=1); a.set_title(f"Family {fam} (n={d.instance.nunique()})")
        if fam == "S":
            a.set_ylabel("% above best realised cost on instance")
    axs[0].legend([bp["boxes"][0], bp2["boxes"][0]], ["mean (filled)", "CVaR95 (hatched)"], loc="upper left")
    save(fig, "fig3_realised_cost_distribution")


def fig_scal(inst):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ["P1-SLA-CO-MILP", "B2-Det-MILP", "P2-SLA-ALNS", "B3-GA", "B4-SA"]:
        d = inst[inst.method == m]
        axs[0].scatter(d.n_sys, d.time, color=COL[m], marker=MK[m], s=18, label=m, alpha=0.8)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("Number of systems"); axs[0].set_ylabel("Wall-clock time (s)")
    axs[0].legend(); axs[0].set_title("(a) runtime (metaheuristics are time-budgeted; MILP capped at time limit)")
    p1 = inst[inst.method == "P1-SLA-CO-MILP"].set_index("instance")
    for m in ["P2-SLA-ALNS", "B3-GA", "B4-SA", "B1-Industry"]:
        d = inst[inst.method == m].set_index("instance")
        common = d.index.intersection(p1.index)
        y = 100 * (d.loc[common, "expected_cost"] - p1.loc[common, "expected_cost"]) / p1.loc[common, "expected_cost"]
        axs[1].scatter(d.loc[common, "n_sys"], y, color=COL[m], marker=MK[m], s=18, label=f"{m} vs P1")
    axs[1].scatter(p1.n_sys, 100 * p1.gap, color=COL["P1-SLA-CO-MILP"], marker="s", s=18, label="P1 optimality gap")
    axs[1].set_xscale("log"); axs[1].set_yscale("symlog", linthresh=0.1); axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_xlabel("Number of systems"); axs[1].set_ylabel("% (expected cost)"); axs[1].legend(fontsize=7)
    axs[1].set_title("(b) expected-cost deviation from P1 incumbent, and P1 gap")
    save(fig, "fig4_scalability")


def fig_conv():
    name = "M05" if os.path.exists(os.path.join(R, "histories", "alns_M05.json")) else "M00"
    a = json.load(open(os.path.join(R, "histories", f"alns_{name}.json")))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    h = np.array(a["hist"])
    axs[0].plot(h[:, 1], h[:, 3] / 1e6, color=COL["P2-SLA-ALNS"], label="P2 ALNS best")
    axs[0].plot(h[:, 1], h[:, 2] / 1e6, color=COL["P2-SLA-ALNS"], alpha=0.3, lw=0.7, label="P2 ALNS current")
    for k, m in [("ga", "B3-GA"), ("sa", "B4-SA")]:
        f = os.path.join(R, "histories", f"{k}_{name}.json")
        if os.path.exists(f):
            hh = np.array(json.load(open(f))["hist"])
            if len(hh):
                y = hh[:, 2] / 1e6
                axs[0].plot(hh[:, 1], np.where(y < 50, y, np.nan), color=COL[m], label=m + " best (penalised fitness)")
    allm = pd.read_csv(os.path.join(R, "ALL_main_runs.csv"))
    p1 = allm[(allm.instance == name) & (allm.method == "P1-SLA-CO-MILP")]
    if len(p1):
        axs[0].axhline(p1.expected_cost.iloc[0] / 1e6, color=COL["P1-SLA-CO-MILP"], ls="--", label="P1 (MILP) objective")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("expected cost (M USD)"); axs[0].set_title(f"(a) convergence, instance {name}, seed 0")
    lo = p1.expected_cost.iloc[0] / 1e6 if len(p1) else h[:, 3].min() / 1e6
    axs[0].set_ylim(lo * 0.995, max(h[0, 2] / 1e6, lo) * 1.03); axs[0].legend(fontsize=7)
    w = pd.DataFrame(a["whist"])
    for c in [c for c in w.columns if c.startswith("d_")]:
        axs[1].plot(w.it, w[c], label=c[2:])
    axs[1].set_title("(b) destroy-operator weights (normalised)"); axs[1].set_xlabel("iteration"); axs[1].legend(fontsize=7)
    for c in [c for c in w.columns if c.startswith("r_")]:
        axs[2].plot(w.it, w[c], label=c[2:])
    axs[2].set_title("(c) repair-operator weights (normalised)"); axs[2].set_xlabel("iteration"); axs[2].legend(fontsize=7)
    save(fig, "fig5_alns_convergence_weights")


def sens_tables():
    f = os.path.join(R, "raw", "sensitivity.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f); d.to_csv(os.path.join(R, "ALL_sensitivity.csv"), index=False)
    d["fam"] = d.instance.str[0]
    agg = d.groupby(["factor", "value", "fam", "method"]).agg(n=("instance", "count"), solved=("plan_cost", lambda x: x.notna().sum()),
                                                              cc_feas=("cc_feasible", "mean"),
                                                              plan_cost_M=("plan_cost", lambda x: x.mean() / 1e6),
                                                              exp_cost_M=("expected_cost", lambda x: x.mean() / 1e6),
                                                              real_cost_M=("sim_cost_mean", lambda x: x.mean() / 1e6),
                                                              cvar95_M=("sim_cvar95", lambda x: x.mean() / 1e6),
                                                              prd_viol_pct=("sim_prd_ev_viol_rate", lambda x: 100 * x.mean()),
                                                              month_viol_pct=("sim_month_viol_rate", lambda x: 100 * x.mean()),
                                                              makespan=("makespan", "mean")).reset_index()
    agg.to_csv(os.path.join(R, "ALL_sensitivity_summary.csv"), index=False)
    OUT.append("### Sensitivity (P1 vs B2; S family unless fam=M; 15 s/30 s limits)\n\nNote: solved = instances with a feasible plan; costs averaged over solved instances only.\n\n" + md(agg, ".4g") + "\n")
    e = agg[(agg.factor == "eps")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for fam, ls in [("S", "-"), ("M", "--")]:
        for m in ["P1-SLA-CO-MILP", "B2-Det-MILP"]:
            x = e[(e.fam == fam) & (e.method == m)].sort_values("value")
            axs[0].plot(x.value, x.plan_cost_M, ls, marker=MK[m], color=COL[m], label=f"{m} ({fam})")
            axs[1].plot(x.value, x.real_cost_M, ls, marker=MK[m], color=COL[m], label=f"{m} ({fam})")
            axs[2].plot(x.value, x.prd_viol_pct, ls, marker=MK[m], color=COL[m], label=f"{m} ({fam})")
    for a, t in zip(axs, ["(a) plan cost (M USD)", "(b) realised mean cost (M USD)", "(c) realised PRD event violation (%)"]):
        a.set_xscale("log"); a.set_xlabel("ε (PRD gold/silver; other classes scaled)"); a.set_title(t)
        a.set_xticks([0.01, 0.05, 0.1, 0.2]); a.set_xticklabels(["0.01", "0.05", "0.10", "0.20"])
    axs[2].legend(fontsize=7)
    save(fig, "fig6_sensitivity_eps")


def ablation_table():
    f = os.path.join(R, "raw", "ablation.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f); d.to_csv(os.path.join(R, "ALL_ablation.csv"), index=False)
    full = d[d.variant == "full"].set_index(["instance", "seed"])
    rows = []
    for v in d.variant.unique():
        x = d[d.variant == v].set_index(["instance", "seed"])
        rel = 100 * (x.expected_cost - full.expected_cost) / full.expected_cost
        relr = 100 * (x.sim_cost_mean - full.sim_cost_mean) / full.sim_cost_mean
        p = stats.wilcoxon(x.sim_cost_mean, full.sim_cost_mean).pvalue if v != "full" and (x.sim_cost_mean != full.sim_cost_mean).any() else np.nan
        rows.append(dict(variant=v, n_runs=len(x), iters=x.iters.mean(), exp_cost_M=x.expected_cost.mean() / 1e6, d_exp_vs_full_pct=rel.mean(),
                         real_cost_M=x.sim_cost_mean.mean() / 1e6, d_real_vs_full_pct=relr.mean(), wilcoxon_p_real=p,
                         cc_feas_pct=100 * x.cc_feasible.mean(), prd_viol_pct=100 * x.sim_prd_ev_viol_rate.mean(),
                         month_viol_pct=100 * x.sim_month_viol_rate.mean()))
    T = pd.DataFrame(rows)
    OUT.append("### Ablation of P2 (M00-M04, seeds 0-1, 8 s each)\n\n" + md(T, ".4g") + "\n")


def pareto_fig():
    f = os.path.join(R, "raw", "pareto.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f); pts = pd.read_csv(os.path.join(R, "raw", "pareto_points.csv"))
    d.to_csv(os.path.join(R, "ALL_pareto_metrics.csv"), index=False); pts.to_csv(os.path.join(R, "ALL_pareto_points.csv"), index=False)
    OUT.append("### Bi-objective (expected cost vs makespan), S00-S05: exact ε-constraint (P1, 15 s per point) vs NSGA-II (pop 40, 20 s)\n\n"
               "HV computed on objectives normalised by the union of both fronts (ref point 1.1,1.1); IGD w.r.t. exact front.\n\n" + md(d, ".4g") + "\n"
               + f"\nMean HV ratio NSGA-II/exact = {d.hv_ratio.mean():.3f} (sd {d.hv_ratio.std():.3f}); mean IGD = {d.igd.mean():.3f}.\n")
    ins = sorted(d.instance.unique())
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for a, n in zip(axs.flat, ins):
        e = pts[(pts.instance == n) & (pts.method == "Exact-eps")].sort_values("makespan")
        a.step(e.makespan, e.cost / 1e6, where="post", color=COL["P1-SLA-CO-MILP"], marker="s", label="Exact ε-constraint (P1)")
        for s, mk in [(0, "o"), (1, "^")]:
            x = pts[(pts.instance == n) & (pts.method == "NSGA-II") & (pts.seed == s)].sort_values("makespan")
            a.scatter(x.makespan, x.cost / 1e6, color="#E69F00", marker=mk, label=f"NSGA-II seed {s}", zorder=3)
        a.set_title(n); a.set_xlabel("makespan (windows)"); a.set_ylabel("expected cost (M USD)")
    axs.flat[0].legend(fontsize=7)
    save(fig, "fig8_pareto_fronts")


def hard_table():
    f = os.path.join(R, "raw", "hard_coupling.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f); d.to_csv(os.path.join(R, "ALL_hard_coupling.csv"), index=False)
    cols = [c for c in ["instance", "variant", "status", "time", "plan_cost", "expected_cost", "makespan", "tight_edges_separated", "n_tight",
                        "sim_cost_mean", "sim_prd_ev_viol_rate"] if c in d.columns]
    OUT.append("### Hard vs soft coupling (P1)\n\n" + md(d[cols], ".4g") + "\n")


def gantt(name="M05"):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import Problem, load
    from calib import METHODS
    f = os.path.join(R, "plans", f"{name}_P1-SLA-CO-MILP_0.json")
    if not os.path.exists(f):
        name = "M00"; f = os.path.join(R, "plans", f"{name}_P1-SLA-CO-MILP_0.json")
    inst = load(os.path.join(INSTANCES, name + ".json")); P = Problem(inst, "cc")
    plan = json.load(open(f)); mt = plan["mt"]
    mn = list(METHODS)
    mcol = {"BR_HOMO": "#56B4E9", "HSR": "#0072B2", "HETEXP": "#E69F00", "DMO_SM_MIG": "#F0E442", "NZDM_MIG": "#CC79A7",
            "DMO_SM": "#D55E00", "DOC": "#009E73", "NZDT": "#000000", "SUM_STD": "#999999", "SUM_ZDO": "#666666",
            "PATCH_STD": "#BBBBBB", "PATCH_HA": "#444444"}
    order = sorted(range(P.nS), key=lambda i: (P.S[i]["landscape"], ["DEV", "QAS", "PRD"].index(P.S[i]["tier"])))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.16 * P.nS)))
    ypos = {i: k for k, i in enumerate(order)}
    used = set()
    for t in range(1, P.T + 1):
        if inst["windows"][t - 1]["blackout_fin"]:
            ax.axvspan(t - 0.5, t + 0.5, color="#F0E442", alpha=0.15, lw=0)
    ax.axvline(P.Td + 0.5, color="red", ls="--", lw=1)
    for j, v in enumerate(mt):
        if v is None:
            continue
        m, t = v; i = P.job_sys[j]; name_m = mn[m]; used.add(name_m)
        if P.job_trans[j]:
            ax.barh(ypos[i], 0.9, left=t - 0.45, color=mcol[name_m], edgecolor="k", lw=0.3)
            ax.plot([t + 0.45, P.T + 0.5], [ypos[i], ypos[i]], color="#0072B2", alpha=0.15, lw=3) if P.migr[i] else None
        else:
            ax.scatter(t, ypos[i], marker="|", s=40, color=mcol[name_m])
    ax.set_yticks(range(P.nS)); ax.set_yticklabels([f"{P.S[i]['name']} [{P.S[i]['sla']}]" for i in order], fontsize=5)
    ax.set_xlabel("maintenance window (week)"); ax.set_xlim(0.5, P.T + 0.5); ax.invert_yaxis()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hl = [Patch(color=mcol[m], label=m) for m in mn if m in used]
    hl += [Line2D([0], [0], color="#0072B2", alpha=0.3, lw=3, label="running on hyperscaler"),
           Patch(color="#F0E442", alpha=0.3, label="finance blackout (PRD ERP/S4/BW)"), Line2D([0], [0], color="red", ls="--", label="ERP deadline T_d")]
    ax.legend(handles=hl, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.set_title(f"P1 SLA-CO plan for {name}: transitions (bars, colour = method) and maintenance (ticks)")
    save(fig, "fig7_gantt_plan")


def instance_stats():
    rows = []
    for f in sorted(glob.glob(os.path.join(INSTANCES, "*.json"))):
        I = json.load(open(f)); S = I["systems"]
        rows.append(dict(instance=I["name"], family=I["family"], systems=len(S), landscapes=len(set(s["landscape"] for s in S)),
                         prd_gold=sum(s["sla"] == "gold" and s["tier"] == "PRD" for s in S), anyDB_share=np.mean([s["src"] == "anyDB" for s in S]),
                         total_db_TB=sum(s["db"] for s in S) / 1000, edges=len(I["edges"]), tight=sum(e["kind"] == "tight" for e in I["edges"]),
                         T=I["T"], Td=I["Td"], C=I["windows"][0]["C"], B_gbps=I["windows"][0]["B"], Pmax=I["windows"][0]["P"]))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(R, "ALL_instance_stats.csv"), index=False)
    a = d.groupby("family").agg(n=("instance", "count"), systems=("systems", lambda x: f"{x.min()}-{x.max()}"),
                                prd_gold=("prd_gold", "mean"), anyDB_share=("anyDB_share", "mean"), total_db_TB=("total_db_TB", "mean"),
                                edges=("edges", "mean"), tight=("tight", "mean"), C=("C", "mean"), B_gbps=("B_gbps", "mean"), Pmax=("Pmax", "mean")).reindex(FAMS).reset_index()
    OUT.append("### Instance statistics (family means)\n\n" + md(a, ".3g") + "\n")


def calib_tables():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import calib as C
    sk = pd.DataFrame([dict(hyperscaler=k[0], sku=k[1], vCPU=k[2], mem_GB=k[3], SAPS=k[4], usd_per_h=k[5], source=k[6].split(" (")[0]) for k in C.SKUS])
    me = pd.DataFrame([dict(method=m, job_type=v["jtype"], setup_h=v["setup_h"], thr_GBh=v["thr"], cv=v["cv"], labour_ph=v["labour"],
                            service_usd=v["service"], p_fail=v["pfail"], rollback_h=v["rollback_h"], replication=v["repl"],
                            wan_frac=v["transfer"], presync_wk=v["presync"], prereq=v["prereq"], prd_only=v["prd_only"], source=v["src"]) for m, v in C.METHODS.items()])
    sl = pd.DataFrame([dict(sla_class=k, D_h=v["D"], A=v["A"], budget_h_per_month=round((1 - v["A"]) * C.MONTH_HOURS, 2), eps=v["eps"], penalty_usd=v["pen"]) for k, v in C.SLA.items()])
    pr = pd.DataFrame([dict(product=k, prd_db_median_GB=v["db"], prd_SAPS=f"{v['saps'][0]}-{v['saps'][1]}", p_anyDB=v["p_any"], job_type=v["jtype"],
                            prd_sla=v["sla"], finance_blackout=v["fin"], mix_weight=v["w"]) for k, v in C.PRODUCTS.items()])
    misc = pd.DataFrame([
        dict(parameter="ECC deadline T_d (parameterised, window 22 of 26)", value="end-2027 mainstream maintenance motivates T_d", source=C.SRC_ECC),
        dict(parameter="labour rate", value=f"{C.LABOUR_RATE} USD/ph", source="ASSUMPTION"),
        dict(parameter="on-prem weekly TCO factor", value=f"{C.ONPREM_FACTOR} x cheapest cloud on-demand", source="ASSUMPTION"),
        dict(parameter="non-prod VM utilisation", value=C.NONPROD_UTIL, source="ASSUMPTION"),
        dict(parameter="tier labour factor", value=str(C.TIER_LABOUR), source="ASSUMPTION"),
        dict(parameter="hybrid-operation penalty per week separation", value=str(C.HYBRID_PEN), source="ASSUMPTION"),
        dict(parameter="deadline miss penalty", value=C.DEADLINE_MISS_PEN, source="ASSUMPTION"),
        dict(parameter="WAN efficiency", value=C.WAN_EFF, source="ASSUMPTION"),
        dict(parameter="failure progress kappa", value=C.FAIL_KAPPA, source="ASSUMPTION"),
        dict(parameter="target HANA RAM per GB source DB", value=str(C.MEM_FACTOR), source="ASSUMPTION"),
        dict(parameter="window length", value="48 h (72 h every 9th window)", source="ASSUMPTION"),
        dict(parameter="capacity C_t, bandwidth B_t, max parallel cutovers", value="generated relative to workload (see gen.py)", source="ASSUMPTION"),
        dict(parameter="DB sizes / product mix", value="see product table", source=C.PRODUCT_SRC),
        dict(parameter="SLA classes", value="see SLA table", source=C.SLA_SRC),
    ])
    srcs = "\n".join(f"- {s}" for s in [C.SRC_AWS_PRICE, C.SRC_AWS_CERT, C.SRC_GCP, C.SRC_GCP_CERT, C.SRC_AZ_PRICE, C.SRC_AZ_SAPS, C.SRC_ECC, C.SRC_DMO, C.SRC_THR])
    return ("## Calibration (accessed " + C.ACCESS_DATE + ")\n\n### Target SKUs\n\n" + md(sk, ".4g") + "\n\nSource details:\n" + srcs +
            "\n\n### Methods\n\n" + md(me, ".4g") + "\n\n### SLA classes\n\n" + md(sl, ".4g") + "\n\n### Products\n\n" + md(pr, ".4g") +
            "\n\n### Other parameters\n\n" + md(misc) + "\n")


def postsim_table():
    f = os.path.join(R, "ALL_postsim_penalty_stress.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    d["nonpen"] = d.cost_mean - d.pen_mean
    rows = []
    for (fam, sc, m), x in d.groupby(["family", "scenario", "method"]):
        r = dict(family=fam, scenario=sc, method=m, n=len(x), real_cost_M=x.cost_mean.mean() / 1e6, penalty_part_M=x.pen_mean.mean() / 1e6,
                 prd_viol_pct=100 * x.prd_ev_viol_rate.mean(), month_viol_pct=100 * x.month_viol_rate.mean())
        for k in (1, 2, 5, 10):
            r[f"cost_pen_x{k}_M"] = (x.nonpen + k * x.pen_mean).mean() / 1e6
        rows.append(r)
    T = pd.DataFrame(rows)
    # break-even penalty multiplier P1 vs B2 per instance
    be = []
    nom = d[d.scenario == "nominal"]
    for n_, x in nom.groupby("instance"):
        a = x[x.method == "P1-SLA-CO-MILP"]; b = x[x.method == "B2-Det-MILP"]
        if len(a) and len(b):
            dn = b.nonpen.iloc[0] - a.nonpen.iloc[0]; dp = a.pen_mean.iloc[0] - b.pen_mean.iloc[0]
            be.append(dn / dp if dp < 0 else np.nan)
    be = np.array(be, float)
    OUT.append("### Penalty decomposition, penalty-multiplier what-if and misspecification stress (re-simulation of saved seed-0 plans, 4000 scenarios)\n\n"
               "cost_pen_xk = realised non-penalty cost + k x realised SLA/deadline penalties (k scales all penalty rates).\n\n" + md(T, ".4g") +
               f"\n\nBreak-even penalty multiplier at which P1's realised mean cost equals B2's (per instance, S+M): median {np.nanmedian(be):.2f}, "
               f"range [{np.nanmin(be):.2f}, {np.nanmax(be):.2f}], n={np.isfinite(be).sum()}.\n")


if __name__ == "__main__":
    instance_stats()
    df, inst = main_tables()
    statistics(inst)
    fig_tradeoff(inst); fig_costdist(inst); fig_scal(inst)
    try:
        fig_conv()
    except Exception as ex:
        print("conv fig failed", ex)
    sens_tables(); ablation_table(); pareto_fig(); hard_table(); postsim_table()
    open(os.path.join(R, "calibration_autogen.md"), "w").write(calib_tables())
    try:
        gantt()
    except Exception as ex:
        print("gantt failed", ex)
    open(os.path.join(R, "tables_autogen.md"), "w").write("\n".join(OUT))
    print("ok")

