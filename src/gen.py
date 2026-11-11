"""Instance generator for SLA-CO-SAP. Writes JSON to <repo>/instances/ (override with SLACO_INSTANCES)."""
import json, os, sys
import numpy as np
import networkx as nx
from calib import *

FAMILIES = {  # n_landscapes range, n instances, T, Td, base bandwidth Gbps, max parallel cutovers
    "S": dict(nl=(4, 7), n=10, T=26, Td=22),
    "M": dict(nl=(14, 26), n=10, T=26, Td=22),
    "L": dict(nl=(50, 100), n=5, T=26, Td=22),
    "XL": dict(nl=(167, 167), n=2, T=26, Td=22),
}
WEEKDAY_START = 5  # window = weekend starting Saturday; day index of first Saturday of the year


def windows(T, rng):
    W = []
    mstarts = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    for t in range(1, T + 1):
        day = (t - 1) * 7 + WEEKDAY_START  # 0-based day-of-year
        month = int(np.searchsorted(mstarts, day, side="right") - 1)
        dom = day - mstarts[month] + 1
        mlen = mstarts[month + 1] - mstarts[month]
        L = 72.0 if t % 9 == 0 else 48.0  # long (holiday) weekend every 9th window -- ASSUMPTION
        fin_black = bool(dom <= 4 or (month % 3 == 2 and dom >= mlen - 4))  # month-start close, quarter-end
        W.append(dict(t=t, L=L, month=month, quarter=(t - 1) // 13, blackout_fin=fin_black))
    return W


def gen_instance(fam, idx, seed):
    rng = np.random.default_rng(seed)
    F = FAMILIES[fam]
    nl = int(rng.integers(F["nl"][0], F["nl"][1] + 1))
    prods = list(PRODUCTS)
    pw = np.array([PRODUCTS[p]["w"] for p in prods]); pw /= pw.sum()
    lands = ["ERP"] + list(rng.choice(prods, size=nl - 1, p=pw))
    systems = []
    for li, p in enumerate(lands):
        P = PRODUCTS[p]
        db_prd = float(np.clip(P["db"] * np.exp(rng.normal(0, 0.6)), 30, 6000))
        src = "anyDB" if rng.random() < P["p_any"] else "HANA"
        saps_prd = float(rng.uniform(*P["saps"]))
        jtype = P["jtype"]
        if p == "BW" and rng.random() < 0.3:
            jtype = "MIGUPG"
        already_cloud = jtype == "UPGRADE"
        cloud_h = HYPERSCALERS[int(rng.integers(3))] if already_cloud else None
        for tier in ["DEV", "QAS", "PRD"]:
            if tier == "PRD":
                db, saps, sla = db_prd, saps_prd, P["sla"]
            elif tier == "QAS":
                db, saps, sla = db_prd * rng.uniform(0.6, 1.0), saps_prd * 0.5, "nonprod"
            else:
                db, saps, sla = db_prd * rng.uniform(0.05, 0.2), saps_prd * 0.15, "nonprod"
            db = max(db, 20.0)
            mem = float(min(max(32.0, MEM_FACTOR[src] * db), 5700.0))
            systems.append(dict(id=len(systems), name=f"{p}{li}-{tier}", product=p, landscape=li, tier=tier,
                                db=round(db, 1), src=src, saps=round(saps), mem=round(mem), sla=sla,
                                jtype=jtype, already_cloud=already_cloud, cloud_h=cloud_h,
                                fin=bool(P["fin"] and tier == "PRD"),
                                w={"gold": 3, "silver": 2, "bronze": 1, "nonprod": 0.5}[sla]))
    # interface graph among PRD systems: hub-and-spoke around ERP PRDs + random loose edges
    prd = [s for s in systems if s["tier"] == "PRD"]
    hubs = [s for s in prd if s["product"] == "ERP"]
    edges = []
    for s in prd:
        if s["product"] == "ERP":
            continue
        if rng.random() < 0.8:
            h = hubs[int(rng.integers(len(hubs)))]
            tight = (s["product"] in ("PIPO", "BW", "CRM")) and rng.random() < 0.5 and not s["already_cloud"]
            edges.append(dict(a=h["id"], b=s["id"], kind="tight" if tight else "loose"))
    ids = [s["id"] for s in prd]
    for i in range(len(ids)):
        for k in range(i + 1, len(ids)):
            if rng.random() < 0.03:
                if not any((e["a"], e["b"]) in [(ids[i], ids[k]), (ids[k], ids[i])] for e in edges):
                    edges.append(dict(a=ids[i], b=ids[k], kind="loose"))
    lags = {li: int(rng.integers(1, 3)) for li in range(nl)}
    W = windows(F["T"], rng)
    # capacity calibration relative to workload (ASSUMPTION: makes resources bind)
    tot_eff = 0.0; tot_vol = 0.0
    for s in systems:
        tot_eff += 150 * TIER_LABOUR[s["tier"]]
        if not s["already_cloud"]:
            tot_vol += s["db"]
    Td = F["Td"]
    C = max(1.6 * tot_eff / (Td - 4), 400.0)
    vol_cap_per_window = 2.0 * tot_vol / (Td - 4)
    max_single = max(s["db"] for s in systems if not s["already_cloud"])
    vol_cap_per_window = max(vol_cap_per_window, 1.2 * max_single)
    B = vol_cap_per_window / (GB_PER_GBPS_HOUR * 48 * WAN_EFF)
    Pmax = max(3, int(np.ceil(2.0 * len(systems) / (Td - 4))))
    for w in W:
        w["C"] = round(C); w["B"] = round(B, 3); w["P"] = Pmax
    inst = dict(name=f"{fam}{idx:02d}", family=fam, seed=seed, T=F["T"], Td=Td, systems=systems, edges=edges,
                lags=lags, windows=W, maint_per_quarter=1)
    return inst


def main():
    from paths import INSTANCES as out
    os.makedirs(out, exist_ok=True)
    for fi, fam in enumerate(FAMILIES):
        for k in range(FAMILIES[fam]["n"]):
            inst = gen_instance(fam, k, seed=1000 * (fi + 1) + k)
            with open(os.path.join(out, inst["name"] + ".json"), "w") as f:
                json.dump(inst, f)
            print(inst["name"], len(inst["systems"]), len(inst["edges"]))


if __name__ == "__main__":
    main()
