# SLA-CO-SAP: SLA-Constrained Optimization for SAP Upgrades, Migration and Maintenance on Hyperscalers

Code, benchmark instances and results for the paper *"An SLA-Constrained Optimization Framework for SAP Upgrades, Migration, and Maintenance on Hyperscalers (SLA-CO-SAP)"*.

SLA-CO-SAP plans a multi-landscape SAP transition (conversions to S/4HANA, migrations to AWS, Azure or Google Cloud, upgrades and recurring maintenance) over a horizon of weekly maintenance windows. It decides when each job runs, which migration or upgrade method it uses and on which SAP-certified hyperscaler machine each system lands. Cutover downtime is uncertain: it follows a lognormal throughput model mixed with a cutover-failure/rollback component. Each job must meet its SLA downtime limit with probability at least 1 − ε. A closed-form deterministic equivalent of this chance constraint keeps the model a linear MILP. The repository contains an exact CP-SAT formulation (P1), the SLA-ALNS matheuristic (P2) and four baselines: an industry wave heuristic (B1), a deterministic MILP (B2), a genetic algorithm (B3) and simulated annealing (B4). It also contains a Monte Carlo audit of every plan, 27 synthetic instances calibrated from public SAP and hyperscaler sources, and the statistical evaluation used in the paper (Quade test, Quade post-hoc tests with Finner correction, exact sign tests and Dolan–Moré performance profiles).

## Citation

> [Author names] ([Year]). An SLA-Constrained Optimization Framework for SAP Upgrades, Migration, and Maintenance on Hyperscalers (SLA-CO-SAP). *[Journal]*, [Volume], [Pages]. https://doi.org/[DOI]

See `CITATION.cff` (placeholders to be filled in on publication).

## Repository structure

```
sla-co-sap/
├── README.md  LICENSE  CITATION.cff  requirements.txt  .gitignore
├── instances/            27 benchmark instances (S00–S09, M00–M09, L00–L04, XL00–XL01), JSON
├── src/
│   ├── paths.py          repository-relative data locations (env overrides SLACO_INSTANCES / SLACO_RESULTS)
│   ├── calib.py          every calibration parameter, with its source URL or the label ASSUMPTION
│   ├── gen.py            deterministic instance generator -> instances/
│   ├── model.py          SLA-CO model: chance-constraint deterministic equivalents, plan evaluator, CP-SAT model (P1, B2)
│   ├── heur.py           B1 industry wave heuristic, P2 SLA-ALNS, B3 GA, B4 SA
│   ├── sim.py            Monte Carlo evaluation of fixed plans
│   ├── run_main.py       main experiment (all six methods)
│   ├── run_extra.py      sensitivity, hard coupling, ablation, bi-objective (Pareto) experiments
│   ├── postsim.py        penalty decomposition / misspecification stress on saved plans
│   ├── analysis.py       aggregation, descriptive tables and Figs. 2, 3, 5–8
│   ├── stats_v2.py       STATISTICS ENTRY POINT: Quade, Quade post-hoc + Finner, sign tests, performance profiles (Fig. 4)
│   └── smoke_test.py     quick run of all methods on one instance with reduced budgets
└── results/
    ├── ALL_*.csv         aggregated per-run, per-instance and summary data behind the tables
    ├── tables_autogen.md, calibration_autogen.md   markdown tables written by analysis.py
    ├── figures/          PNG (300 dpi) and PDF figures
    ├── raw/              per-experiment CSVs written by run_main.py / run_extra.py
    ├── plans/            saved plans (method/window/target of every job) for each instance, method and seed
    ├── histories/        convergence histories (seed 0) of P2, GA, SA; ALNS operator weights
    └── versions.txt      package versions recorded when the experiments were run
```

## Environment

Python 3.13 (verified with 3.13.18).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root, e.g. `python src/stats_v2.py`. By default they read `instances/` and read and write `results/`. To write somewhere else, set `SLACO_RESULTS=/path/to/dir` (and `SLACO_INSTANCES` for instances).

## Instances and calibration

`instances/` holds the 27 instances used in the paper. `src/gen.py` regenerates them deterministically. Instance *k* of family *f* (S=1, M=2, L=3, XL=4) uses seed `1000·f + k`. Regenerating them reproduces the shipped files byte for byte:

```bash
SLACO_INSTANCES=/tmp/inst python src/gen.py && diff -r /tmp/inst instances
```

| Family | Instances | Landscapes (generator range) | Systems | Horizon T / ERP deadline T_d |
|---|---|---|---|---|
| S  | 10 | 4–7     | 12–21   | 26 / 22 |
| M  | 10 | 14–26   | 42–75   | 26 / 22 |
| L  | 5  | 50–100  | 198–294 | 26 / 22 |
| XL | 2  | 167     | 501     | 26 / 22 |

Each landscape has DEV, QAS and PRD tiers. Windows are weekend windows of 48 h (72 h every ninth window), with finance blackouts at month start and quarter end. Team capacity, WAN volume and the number of parallel cutovers are set relative to each instance's workload.

`src/calib.py` documents every parameter. Each one carries a source URL (accessed 31 October 2026) or the explicit label `ASSUMPTION`. `analysis.py` prints the calibration tables to `results/calibration_autogen.md`. Sources used:

- AWS SKUs, SAPS and prices: https://aws-pricing.com/sap.html (aggregator of the AWS price list and SAP Note 1656099); certification: https://docs.aws.amazon.com/sap/latest/general/sap-hana-aws-ec2.html
- Google Cloud SKUs, SAPS and prices: https://gcloud-compute.com/sap.html (aggregator, SAP Note 2456432); certification: https://docs.cloud.google.com/sap/docs/certifications-sap-hana
- Azure prices: https://www.azurespeed.com/AzureVmPricing/ and https://costcalc.cloudoptimo.com (Linux PAYG, East US). Azure SAPS: https://fgarofalo56.github.io/csa-inabox/migrations/sap-to-azure/benchmarks/ (secondary community source, because SAP Note 1928533 requires a login)
- Migration throughput bands (DMO, HSR, backup/restore): the same community benchmark page; DMO tuning guidance: https://support.sap.com/en/tools/software-logistics-tools/software-update-manager/dmo_performance_optimization.html
- ECC end of mainstream maintenance (motivates T_d): https://www.theregister.com/2025/01/28/sap_extends_support_deadline/
- These are assumptions: setup times, failure probabilities, labour and service costs, SLA classes and penalties, product mix and DB sizes, WAN efficiency, capacity rules.

## Reproducing the paper

### Full pipeline

The full pipeline took about 55 minutes of wall-clock time on the machine described below.

```bash
python src/gen.py                         # optional: instances/ already shipped
rm -rf results/raw results/plans results/histories   # raw CSVs are APPENDED to, so start clean
python src/run_main.py S M &              # main experiment, process 1
python src/run_main.py L XL &             # main experiment, process 2 (run concurrently, as in the paper)
wait
python src/run_extra.py sens hard         # sensitivity + hard coupling
python src/run_extra.py ablation pareto   # SLA-ALNS ablation + bi-objective study
python src/postsim.py                     # penalty decomposition + misspecification stress
python src/analysis.py                    # ALL_*.csv, tables_autogen.md, calibration_autogen.md, Figs. 2,3,5-8
python src/stats_v2.py                    # Quade / Finner / sign tests / performance profiles (Fig. 4)
```

### Statistics only, from the shipped results

This takes about 4 s.

```bash
python src/stats_v2.py
```

The script reads `results/ALL_main_per_instance.csv` and `results/ALL_ablation.csv`. It writes `ALL_quade.csv`, `ALL_quade_posthoc_finner.csv`, `ALL_sign_finner.csv` and `figures/fig9_performance_profiles.{png,pdf}`, and prints the ablation sign tests and the performance-profile values ρ(1) and ρ(1.01). Expected output over all 27 instances: Quade F = 10.65 (realized cost; df 5, 130; p = 1.4·10⁻⁸), F = 55.3 (PRD violation rate) and F = 6.82 (CVaR95; p = 1.1·10⁻⁵).

`python src/analysis.py` can likewise be rerun on the shipped `results/` alone (about 13 s, no solver calls).

### Statistics: which script the paper uses

`stats_v2.py` produces all the inferential statistics reported in the paper. `analysis.py` still contains an older `statistics()` function: Friedman tests, Nemenyi post-hoc tests, Wilcoxon–Holm tests and CD diagrams. It writes `ALL_friedman.csv`, `ALL_nemenyi_realised_cost.csv`, `ALL_wilcoxon_holm.csv`, `figures/fig9_cd_diagram*.{png,pdf}` and the "Friedman"/"Wilcoxon" sections of `tables_autogen.md`. **The paper does not use these outputs.** They are left out of `results/` but are regenerated whenever `analysis.py` runs. The paper does use every other part of `analysis.py`: the aggregation, the descriptive tables and Figs. 2, 3 and 5–8.

### Map from the paper to the code

| Paper item | Content | Produced by | Output file(s) in `results/` |
|---|---|---|---|
| Tables 1, 2, 3, 11 | literature, classification, notation | — (not computed) | — |
| Table 4 | instance families | `gen.py` → `analysis.py` (`instance_stats`) | `ALL_instance_stats.csv`; `tables_autogen.md` § Instance statistics |
| Table 5 | migration/upgrade/maintenance methods | `calib.py` → `analysis.py` (`calib_tables`) | `calibration_autogen.md` § Methods |
| Table 6 | SLA classes | `calib.py` → `analysis.py` (`calib_tables`) | `calibration_autogen.md` § SLA classes |
| §6.2 text | machine catalog (29 SKUs) | `calib.py` → `analysis.py` | `calibration_autogen.md` § Target SKUs |
| Table 7 | main results | `run_main.py` → `analysis.py` (`main_tables`) | `ALL_main_summary.csv` (also `ALL_main_runs.csv`, `ALL_main_per_instance.csv`); `tables_autogen.md` § Main results |
| §6.7 text | Quade F tests (all instances and per family) | `stats_v2.py` | `ALL_quade.csv` |
| Table 8 | Quade post-hoc P2 vs each method, Finner-adjusted; W/T/L | `stats_v2.py` | `ALL_quade_posthoc_finner.csv` (p, Finner); `ALL_sign_finner.csv` (W/T/L) |
| Table 9 | penalty multiplier k and stress test | `postsim.py` → `analysis.py` (`postsim_table`) | `ALL_postsim_penalty_stress.csv`; `tables_autogen.md` § Penalty decomposition |
| Table 10 | SLA-ALNS ablation | `run_extra.py ablation` → `analysis.py`; sign-test p from `stats_v2.py` (stdout) | `ALL_ablation.csv`; `tables_autogen.md` § Ablation |
| §6.11 text | sensitivity (SLA tightness, bandwidth, capacity, deadline) | `run_extra.py sens` → `analysis.py` | `ALL_sensitivity.csv`, `ALL_sensitivity_summary.csv` |
| §6.11 text | hard vs soft coupling | `run_extra.py hard` → `analysis.py` | `ALL_hard_coupling.csv` |
| §6.11 text | bi-objective fronts, NSGA-II (HV ratio) | `run_extra.py pareto` → `analysis.py` | `ALL_pareto_metrics.csv`, `ALL_pareto_points.csv`; `figures/fig8_pareto_fronts.*` (not shown in the paper) |
| Fig. 1, Algorithm 1 | framework diagram, pseudocode | — (drawn by hand) | — |
| Fig. 2 | plan cost vs realized violation / realized cost | `analysis.py` (`fig_tradeoff`) | `figures/fig2_cost_vs_violation.{png,pdf}` |
| Fig. 3 | realized mean cost and CVaR95 vs best | `analysis.py` (`fig_costdist`) | `figures/fig3_realised_cost_distribution.{png,pdf}` |
| Fig. 4 | performance profiles | `stats_v2.py` | `figures/fig9_performance_profiles.{png,pdf}` |
| Fig. 5 | scalability (time, gap, deviation from P1) | `analysis.py` (`fig_scal`) | `figures/fig4_scalability.{png,pdf}` |
| Fig. 6 | convergence and ALNS operator weights on M05 | `run_main.py` (histories) → `analysis.py` (`fig_conv`) | `figures/fig5_alns_convergence_weights.{png,pdf}` |
| Fig. 7 | effect of ε | `run_extra.py sens` → `analysis.py` (`sens_tables`) | `figures/fig6_sensitivity_eps.{png,pdf}` |
| Fig. 8 | P1 plan for M05 (Gantt) | `run_main.py` (plans) → `analysis.py` (`gantt`) | `figures/fig7_gantt_plan.{png,pdf}` |

The figure file names come from an earlier draft and do not match the final paper numbering. Use the table above.

### Smoke test

The smoke test takes about 10 s.

```bash
python src/smoke_test.py S00            # all six methods on S00; 5 s exact / 2 s heuristic, 1 seed, 500 MC scenarios
```

It writes to a fresh temporary directory, never to `results/`. With these short limits, B1, P1 and B2 reproduce the plan costs in `results/raw/main_S.csv` exactly. P1 solves S00 to optimality within the limit, and B1 is deterministic.

## Seeds and time limits

- Instances: seed `1000·f + k` (see above).
- Stochastic methods (P2, GA, SA): seeds 0–4 (S), 0–2 (M, L) and 0–1 (XL). Results are averaged per instance before the statistical tests.
- CP-SAT (P1, B2): `random_seed = 0` and 1 worker. P1 and B2 are warm-started from the B1 plan. The CP-SAT sub-solves inside SLA-ALNS draw their seeds from the ALNS RNG.
- Monte Carlo audit: seed 777 for every plan. It uses 5,000 scenarios (S, M) or 2,000 (L, XL); 3,000 in `run_extra.py`; 4,000 in `postsim.py`.

| Family | P1/B2 CP-SAT time limit | P2/GA/SA time budget | ALNS sub-solve limit |
|---|---|---|---|
| S  | 15 s | 4 s  | 0.8 s |
| M  | 40 s | 8 s  | 1.6 s |
| L  | 60 s | 20 s | 2 s |
| XL | 90 s | 30 s | 2 s |

The extra experiments use these settings:

- Sensitivity and hard coupling: CP-SAT limits of 15 s (S) and 30 s (M).
- Ablation: M00–M04, seeds 0–1, 8 s per run.
- Pareto: S00–S05. The exact ε-constraint fronts use 15 s per solve; NSGA-II uses population 40, a 20 s budget and seeds 0–1.

All limits are wall-clock limits.

## Hardware and runtime used

- **Machine:** a shared VM with 2 vCPUs (Intel Xeon @ 2.10 GHz) and 7 GB RAM, running Linux. CP-SAT used 1 worker.
- **Software:** Python 3.13.18 and OR-Tools 9.15.6755. All recorded versions are in `results/versions.txt`.
- **Concurrency:** `run_main.py S M` and `run_main.py L XL` ran concurrently. From about 30 minutes in, a third process ran the sensitivity experiments. Reported times therefore include CPU contention.
- **Runtime:** the progress logs carry no timings, so these figures come from the file timestamps of the original run.

| Stage | Wall-clock time |
|---|---|
| Instance generation | under 1 s |
| `run_main.py S M` | about 42 min |
| `run_main.py L XL`, concurrently | about 44 min |
| Sensitivity, hard coupling, ablation and Pareto, partly overlapping | finished about 11 min after the main runs |
| Whole pipeline, including `analysis.py` | about 55 min |

## Reproduction caveats

- Time limits are wall-clock limits, and P2, GA, SA and the time-limited CP-SAT runs (M, L, XL) depend on the speed of the machine and its load. Rerunning the experiments therefore gives close but not bit-identical results. The statistics and figures reproduce exactly from the shipped `results/`.
- `results/versions.txt` records pandas 3.0.5 for the original run, but this repository was verified with pandas 2.3.3, the version pinned here. All CSV outputs regenerate identically. The only difference is cosmetic: empty cells print as `nan` in one column of `calibration_autogen.md`.
- `run_main.py` and `run_extra.py` append to `results/raw/*.csv`. Delete the old files before a full rerun, or you will get duplicated rows.
- The Finner-adjusted p-value of P2 vs B1 on PRD violation underflows to 0 in `ALL_quade_posthoc_finner.csv`. The paper reports it as < 10⁻¹⁰.
- The instances are synthetic. Several cost and duration parameters are explicit assumptions; see `src/calib.py`.

## License

MIT; see `LICENSE`.
