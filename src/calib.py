"""Calibration data for SLA-CO-SAP benchmark.

Every parameter carries a 'src' field: either a URL (accessed 2026-10-31) or the
literal label 'ASSUMPTION: <justification>'.  Nothing here is presented as a
sourced fact unless a URL is given.
"""
ACCESS_DATE = "2026-10-31"

SRC_AWS_PRICE = "https://aws-pricing.com/sap.html (third-party aggregator of AWS price list + SAP Note 1656099 SAPS; price column = average across regions, Linux on-demand; region not stated by page)"
SRC_AWS_CERT = "https://docs.aws.amazon.com/sap/latest/general/sap-hana-aws-ec2.html"
SRC_GCP = "https://gcloud-compute.com/sap.html (third-party aggregator of GCP price list + SAP Note 2456432 SAPS; price = average region, on-demand)"
SRC_GCP_CERT = "https://docs.cloud.google.com/sap/docs/certifications-sap-hana"
SRC_AZ_PRICE = "https://www.azurespeed.com/AzureVmPricing/<size> and https://costcalc.cloudoptimo.com (Linux PAYG, East US)"
SRC_AZ_SAPS = "https://fgarofalo56.github.io/csa-inabox/migrations/sap-to-azure/benchmarks/ (community page; SAP Note 1928533 itself is login-protected and could not be accessed -> SAPS values are UNVERIFIED secondary data)"
SRC_ECC = "https://www.theregister.com/2025/01/28/sap_extends_support_deadline/ (mainstream maintenance SAP Business Suite 7 ends end-2027; extended to 2030)"
SRC_DMO = "https://support.sap.com/en/tools/software-logistics-tools/software-update-manager/dmo_performance_optimization.html (DMO tuning guidance, qualitative)"
SRC_THR = "https://fgarofalo56.github.io/csa-inabox/migrations/sap-to-azure/benchmarks/ (community page, unverified: DMO anyDB->HANA 200-450 GB/h; HSR initial copy 500-800 GB/h; backup/restore 300-600 GB/h)"

# ------------------------------------------------------------------ SKUs
# (hyperscaler, name, vCPU, mem_GB, SAPS, usd_per_hour, src)
SKUS = [
    # AWS (SAPS + price from aws-pricing.com)
    ("AWS", "r6i.large", 2, 16, 3063, 0.148, SRC_AWS_PRICE),
    ("AWS", "r6i.xlarge", 4, 32, 6127, 0.296, SRC_AWS_PRICE),
    ("AWS", "r7i.2xlarge", 8, 64, 16620, 0.6156, SRC_AWS_PRICE),
    ("AWS", "r6i.4xlarge", 16, 128, 24506, 1.184, SRC_AWS_PRICE),
    ("AWS", "r6i.16xlarge", 64, 512, 98025, 4.7362, SRC_AWS_PRICE),
    ("AWS", "r7i.24xlarge", 96, 768, 158250, 7.3877, SRC_AWS_PRICE),
    ("AWS", "r6i.32xlarge", 128, 1024, 196050, 9.4735, SRC_AWS_PRICE),
    ("AWS", "x2idn.24xlarge", 96, 1536, 147038, 12.9978, SRC_AWS_PRICE),
    ("AWS", "x2iedn.16xlarge", 64, 2048, 94500, 17.1975, SRC_AWS_PRICE),
    ("AWS", "x2iedn.24xlarge", 96, 3072, 141750, 25.7962, SRC_AWS_PRICE),
    # Azure (price azurespeed/cloudoptimo East US; SAPS community page)
    ("Azure", "E16ds_v5", 16, 128, 21350, 1.152, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "M32ts", 32, 192, 36826, 2.707, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "E32ds_v5", 32, 256, 42700, 2.304, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "E64ds_v5", 64, 512, 85400, 4.608, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "E96ds_v5", 96, 672, 128100, 6.912, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "M64s", 64, 1024, 71502, 6.669, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "M128s", 128, 2048, 143480, 13.338, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "M208s_v2", 208, 2850, 260000, 22.31, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    ("Azure", "M416s_v2", 416, 5700, 488000, 49.58, SRC_AZ_PRICE + " | SAPS: " + SRC_AZ_SAPS),
    # GCP (gcloud-compute.com)
    ("GCP", "n2-highmem-2", 2, 16, 3574, 0.159, SRC_GCP),
    ("GCP", "c3-highmem-4", 4, 32, 7840, 0.3077, SRC_GCP),
    ("GCP", "c3-standard-8", 8, 32, 15670, 0.4674, SRC_GCP),
    ("GCP", "n2-highmem-32", 32, 256, 57185, 2.5447, SRC_GCP),
    ("GCP", "n2-highmem-80", 80, 640, 133400, 6.3618, SRC_GCP),
    ("GCP", "n2-highmem-128", 128, 864, 183350, 9.3559, SRC_GCP),
    ("GCP", "m3-megamem-64", 64, 976, 99170, 8.5156, SRC_GCP),
    ("GCP", "c3-highmem-176", 176, 1408, 271550, 13.5373, SRC_GCP),
    ("GCP", "m3-ultramem-128", 128, 3904, 184800, 28.7976, SRC_GCP),
    ("GCP", "m2-ultramem-208", 208, 5888, 223325, 50.9308, SRC_GCP),
]
HYPERSCALERS = ["AWS", "Azure", "GCP"]

# ------------------------------------------------------------------ methods
# setup_h: fixed downtime (h); thr: mean throughput GB/h (lognormal), cv: coefficient of variation;
# labour: person-hours (PRD; scaled by tier factor); service: USD external service/licence;
# pfail: failure probability -> rollback; rollback_h: extra downtime on failure;
# repl: replication-based (source stays intact; failure downtime = setup + rollback, no data phase);
# transfer: fraction of DB volume crossing the WAN inside the cutover window;
# presync: weeks of target dual-running before cutover (replication seeding);
# prereq: 'HANA' (source DB must be HANA), 'anyDB' (source must be non-HANA), None; prd_only.
A = "ASSUMPTION"
METHODS = {
    # ---- MIGRATE (lift & shift to hyperscaler, HANA target)
    "BR_HOMO": dict(jtype="MIGRATE", setup_h=4.0, thr=450, cv=0.30, labour=40, service=5000, pfail=0.03, rollback_h=3.0, repl=False, transfer=1.0, presync=0, prereq="HANA", prd_only=False,
                    src=f"throughput mid of 300-600 GB/h: {SRC_THR}; setup/labour/cost/pfail: {A} (expert-judgement order of magnitude)"),
    "HSR": dict(jtype="MIGRATE", setup_h=1.5, thr=6000, cv=0.50, labour=60, service=15000, pfail=0.02, rollback_h=0.5, repl=True, transfer=0.05, presync=1, prereq="HANA", prd_only=False,
                src=f"HSR initial copy 500-800 GB/h happens before cutover ({SRC_THR}); cutover downtime = takeover + post-steps: {A}"),
    "HETEXP": dict(jtype="MIGRATE", setup_h=6.0, thr=180, cv=0.40, labour=80, service=10000, pfail=0.06, rollback_h=4.0, repl=False, transfer=1.0, presync=0, prereq="anyDB", prd_only=False,
                   src=f"classic R3load heterogeneous export/import; slower than DMO: {A}"),
    "DMO_SM_MIG": dict(jtype="MIGRATE", setup_h=8.0, thr=300, cv=0.35, labour=90, service=15000, pfail=0.05, rollback_h=4.0, repl=False, transfer=1.0, presync=0, prereq="anyDB", prd_only=False,
                       src=f"DMO anyDB->HANA 200-450 GB/h ({SRC_THR}); DMO tuning {SRC_DMO}; other fields {A}"),
    "NZDM_MIG": dict(jtype="MIGRATE", setup_h=2.0, thr=8000, cv=0.50, labour=200, service=120000, pfail=0.03, rollback_h=0.5, repl=True, transfer=0.1, presync=1, prereq=None, prd_only=True,
                     src=f"near-zero-downtime heterogeneous migration via log/trigger-based replication (nZDM/SLT-like); {A}"),
    # ---- MIGRATE+UPGRADE (e.g. ECC -> S/4HANA conversion with move to cloud)
    "DMO_SM": dict(jtype="MIGUPG", setup_h=12.0, thr=250, cv=0.35, labour=160, service=40000, pfail=0.07, rollback_h=6.0, repl=False, transfer=1.0, presync=0, prereq=None, prd_only=False,
                   src=f"DMO with System Move, throughput in 200-450 GB/h band ({SRC_THR}); conversion-specific setup (SUM downtime phases) {A}"),
    "DOC": dict(jtype="MIGUPG", setup_h=8.0, thr=900, cv=0.45, labour=220, service=90000, pfail=0.06, rollback_h=5.0, repl=False, transfer=0.3, presync=0, prereq=None, prd_only=False,
                src=f"downtime-optimised conversion: most data migrated in uptime -> high effective throughput: {A}"),
    "NZDT": dict(jtype="MIGUPG", setup_h=2.0, thr=10000, cv=0.40, labour=320, service=250000, pfail=0.02, rollback_h=0.5, repl=True, transfer=0.05, presync=2, prereq=None, prd_only=True,
                 src=f"near-zero-downtime service (clone+replay); very low downtime, high cost: {A}"),
    # ---- UPGRADE only (system already on hyperscaler)
    "SUM_STD": dict(jtype="UPGRADE", setup_h=10.0, thr=2000, cv=0.30, labour=60, service=8000, pfail=0.04, rollback_h=4.0, repl=False, transfer=0.0, presync=0, prereq=None, prd_only=False,
                    src=f"standard SUM EHP/SP upgrade: {A}"),
    "SUM_ZDO": dict(jtype="UPGRADE", setup_h=1.5, thr=20000, cv=0.40, labour=120, service=30000, pfail=0.03, rollback_h=0.5, repl=True, transfer=0.0, presync=0, prereq="HANA", prd_only=False,
                    src=f"SUM zero-downtime option (bridge subsystem): {A}"),
    # ---- MAINT (OS/DB/kernel patching)
    "PATCH_STD": dict(jtype="MAINT", setup_h=2.5, thr=4000, cv=0.40, labour=8, service=0, pfail=0.02, rollback_h=2.0, repl=False, transfer=0.0, presync=0, prereq=None, prd_only=False,
                      src=f"offline patch incl. backup/restart, DB-size dependent: {A}"),
    "PATCH_HA": dict(jtype="MAINT", setup_h=0.3, thr=100000, cv=0.50, labour=20, service=1000, pfail=0.02, rollback_h=0.5, repl=True, transfer=0.0, presync=0, prereq=None, prd_only=True,
                     src=f"rolling patch with HA/HSR takeover (PRD only, HA pair assumed): {A}"),
}
FAIL_KAPPA = 0.5  # fraction of data phase done before an (offline) failure is detected -- ASSUMPTION

# ------------------------------------------------------------------ SLA classes
# D: max downtime per event (h); A: monthly availability target applied to *planned* downtime;
# eps: chance-constraint risk level; pen: penalty USD per violated event or violated system-month
SLA = {
    "gold":    dict(D=4.0,  A=0.993, eps=0.05, pen=60000),
    "silver":  dict(D=8.0,  A=0.990, eps=0.05, pen=25000),
    "bronze":  dict(D=24.0, A=0.975, eps=0.10, pen=8000),
    "nonprod": dict(D=48.0, A=0.950, eps=0.20, pen=1000),
}
SLA_SRC = (f"{A}: per-event limits 4/8/24 h follow the task statement; availability targets are applied to PLANNED downtime "
           "only and set so that one gold cutover (~4h) plus one patch does not fit in the same month (99.3% -> 5.1 h/month); "
           "penalties are order-of-magnitude business-impact assumptions.")
MONTH_HOURS = 730.0
LABOUR_RATE = 150.0      # USD/person-hour  -- ASSUMPTION (blended SAP Basis consultant rate)
ONPREM_FACTOR = 1.25     # on-prem weekly TCO = 1.25 x cheapest cloud on-demand cost -- ASSUMPTION
NONPROD_UTIL = 0.5       # non-prod VMs run 50% of hours (snoozing) -- ASSUMPTION
TIER_LABOUR = {"DEV": 0.5, "QAS": 0.7, "PRD": 1.0}  # ASSUMPTION
HYBRID_PEN = {"tight": 5000.0, "loose": 500.0}       # USD per week of separation -- ASSUMPTION
DEADLINE_MISS_PEN = 200000.0                          # USD if ERP transition realised after T_d -- ASSUMPTION
WAN_EFF = 0.7            # usable fraction of nominal WAN bandwidth -- ASSUMPTION
GB_PER_GBPS_HOUR = 450.0  # 1 Gbps = 0.125 GB/s = 450 GB/h (unit conversion)

# product catalogue: PRD DB size median GB (lognormal sigma 0.6), PRD SAPS range, P(anyDB source), job type, PRD SLA, finance (blackout)
PRODUCTS = {
    "ERP":    dict(db=3000, saps=(40000, 150000), p_any=0.6, jtype="MIGUPG", sla="gold", fin=True, w=0.22),
    "S4":     dict(db=1500, saps=(30000, 120000), p_any=0.0, jtype="UPGRADE", sla="gold", fin=True, w=0.06),
    "BW":     dict(db=2000, saps=(20000, 80000), p_any=0.5, jtype="MIGRATE", sla="silver", fin=True, w=0.14),
    "CRM":    dict(db=800, saps=(15000, 60000), p_any=0.6, jtype="MIGRATE", sla="silver", fin=False, w=0.10),
    "SRM":    dict(db=400, saps=(8000, 30000), p_any=0.6, jtype="MIGRATE", sla="silver", fin=False, w=0.08),
    "PIPO":   dict(db=150, saps=(10000, 40000), p_any=0.7, jtype="MIGRATE", sla="gold", fin=False, w=0.12),
    "SOLMAN": dict(db=200, saps=(5000, 15000), p_any=0.7, jtype="MIGRATE", sla="bronze", fin=False, w=0.10),
    "GRC":    dict(db=100, saps=(5000, 15000), p_any=0.7, jtype="MIGRATE", sla="silver", fin=False, w=0.08),
    "PORTAL": dict(db=80, saps=(5000, 20000), p_any=0.8, jtype="MIGRATE", sla="bronze", fin=False, w=0.10),
}
PRODUCT_SRC = f"{A}: product mix, DB-size medians (lognormal sigma=0.6, clipped 30-6000 GB) and SAPS ranges are plausible mid-size-enterprise values, not fitted to a public dataset."
MEM_FACTOR = {"HANA": 1.2, "anyDB": 0.5}  # target HANA RAM per GB source DB (anyDB compresses ~4x, RAM ~2x footprint) -- ASSUMPTION
