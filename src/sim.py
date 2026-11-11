"""Monte Carlo evaluation of a fixed plan under stochastic throughput and failures (same distributions as the model;
optional 'stress' multiplies the throughput CV and failure probabilities to test misspecification)."""
import numpy as np
from calib import *
from model import lognorm_params

MAX_ATTEMPTS = 3  # third attempt assumed to succeed (no further failure) -- ASSUMPTION


def simulate(P, sel, hsel, n=5000, seed=12345, cv_mult=1.0, pf_mult=1.0):
    rng = np.random.default_rng(seed)
    T = P.T; nS = P.nS
    sel = np.asarray(sel)
    mnames = P.mnames
    Lx = np.concatenate([P.L, np.full(MAX_ATTEMPTS + 1, 48.0)])
    monx = np.concatenate([P.month, np.full(MAX_ATTEMPTS + 1, P.nM)])
    nMx = P.nM + 1
    md = np.zeros((n, nS, nMx), np.float32)
    cost = np.zeros(n); penc = np.zeros(n)
    n_events = np.zeros(n); n_viol = np.zeros(n); n_prd_events = np.zeros(n); n_prd_viol = np.zeros(n)
    tau = np.zeros((n, nS), int)
    for j, g in enumerate(sel):
        if g < 0:
            continue
        i = P.O_sys[g]; t0 = P.O_t[g]; M = METHODS[mnames[P.O_m[g]]]; s = P.S[i]
        V = s["db"]; mu, sig = lognorm_params(M["thr"], M["cv"] * cv_mult)
        p = min(1.0, M["pfail"] * pf_mult)
        active = np.ones(n, bool); att = np.zeros(n, int)
        prd = s["tier"] == "PRD"
        base_cost = P.O_service[g] + P.O_eff[g] * LABOUR_RATE
        cost += base_cost
        for k in range(MAX_ATTEMPTS):
            if not active.any():
                break
            th = np.exp(rng.normal(mu, sig, n))
            fail = (rng.random(n) < p) if k < MAX_ATTEMPTS - 1 else np.zeros(n, bool)
            if M["repl"]:
                dfail = np.full(n, M["setup_h"] + M["rollback_h"])
            else:
                dfail = M["setup_h"] + FAIL_KAPPA * V / th + M["rollback_h"]
            down = np.where(fail, dfail, M["setup_h"] + V / th)
            tt = t0 + k
            Dl = min(P.D[i], Lx[tt])
            v = (down > Dl) & active
            n_events += active; n_viol += v
            if prd:
                n_prd_events += active; n_prd_viol += v
            cost += v * P.pen[i]; penc += v * P.pen[i]
            md[active, i, monx[tt]] += down[active].astype(np.float32)
            # failure: rollback + re-run cost (0.5 service + labour)
            fa = fail & active
            cost += fa * (0.5 * P.O_service[g] + P.O_eff[g] * LABOUR_RATE)
            att = np.where(active, k, att)
            active = fa
        if P.O_trans[g]:
            tau[:, i] = t0 + att
    # monthly availability violations (system-months with planned downtime)
    has = md > 0
    mv = (md > P.budget[None, :, None]) & has
    n_sm = has.sum(axis=(1, 2)); n_mv = mv.sum(axis=(1, 2))
    mpen = (mv * P.pen[None, :, None]).sum(axis=(1, 2)); cost += mpen; penc += mpen
    # run costs with realised cutover windows
    hs = hsel[P.comp]
    cw = P.cloud_week[np.arange(nS), hs]
    cw = np.where(np.isfinite(cw), cw, 0.0)
    migr = P.migr
    tcap = np.minimum(tau, T)
    run = (P.onprem_week[None, :] * tcap + cw[None, :] * (T - tcap)) * migr[None, :]
    cost += run.sum(axis=1)
    # presync dual-running (deterministic, part of nominal option cost)
    for j, g in enumerate(sel):
        if g >= 0 and P.O_trans[g] and migr[P.O_sys[g]]:
            M = METHODS[mnames[P.O_m[g]]]
            cost += M["presync"] * 168 * P.onprem_week[P.O_sys[g]] / (ONPREM_FACTOR * 168)
    # hybrid coupling penalty (realised)
    if len(P.E_a):
        dd = np.abs(tau[:, P.E_a] - tau[:, P.E_b])
        if P.hard:
            cost += (dd * P.E_pen * ~P.E_tight).sum(axis=1)
        else:
            cost += (dd * P.E_pen).sum(axis=1)
    # deadline
    erp = np.array([s["product"] == "ERP" for s in P.S])
    miss = (tau[:, erp] > P.Td).any(axis=1)
    cost += miss * DEADLINE_MISS_PEN; penc += miss * DEADLINE_MISS_PEN
    q95 = np.quantile(cost, 0.95)
    return dict(pen_mean=penc.mean(), cost_mean=cost.mean(), cost_sd=cost.std(), cvar95=cost[cost >= q95].mean(),
                ev_viol_rate=(n_viol.sum() / n_events.sum()), prd_ev_viol_rate=(n_prd_viol.sum() / max(n_prd_events.sum(), 1)),
                month_viol_rate=n_mv.sum() / max(n_sm.sum(), 1), p_any_viol=float(((n_viol + n_mv) > 0).mean()),
                makespan_mean=tau.max(axis=1).mean(), deadline_miss=miss.mean(), n_scen=n)
