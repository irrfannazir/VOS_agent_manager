# DOCUMENT 2 — 3-Month Implementation Proposal

**Project:** AOS — Adaptive Cognitive AI Microkernel
**Document type:** Engineering Roadmap (v1.0)
**Delivery model:** Strict incremental — Baseline → Integration → Advanced Novelty. Each month ends in a runnable, measured system.

---

## 0. Ground Rules

1. **Freeze the baseline.** The Month-1 naive system is the control group for all three papers. After M1-EXIT it is tagged, never modified — only compared against.
2. **Telemetry before features.** No module merges without OpenTelemetry spans on its decisions.
3. **Local-first economics.** Development runs use Ollama-served local models exclusively; cloud models appear only in final measured runs. All resource responses go through a record/replay cache so repeated experiments are free and deterministic.
4. **Weekly checkpoint.** Every Friday: run the 20-task smoke suite, record success/cost/latency, commit the numbers. Trend lines are the early-warning system.
5. **Ablation switches from birth.** Every novel module ships with its `--no-X` flag the day it merges.

---

## MONTH 1 — Baseline Kernel (Modules M1–M3)

### Week-by-week

| Week | Build | Validate |
|---|---|---|
| W1 | Repo, CI, OpenTelemetry/Prometheus/Grafana stack, Redis, Postgres, Qdrant via docker-compose. Resource contract (`describe`/`invoke`) + Capability Registry. LiteLLM adapter wrapping 2 cloud LLMs + 2 local (Ollama) models. | Contract unit tests; registry lists 4 model resources with manifests; heartbeat marks a stopped Ollama container `down` within 10 s. |
| W2 | MCP client adapter; wrap 1 toolset (web search + file ops + calculator), 1 vision model, 1 ASR model, 1 database resource, 1 mocked sensor endpoint. Registry `find(flags)` query. | ≥9 heterogeneous resources registered; `find()` returns correct candidates for 20 hand-written capability queries. |
| W3 | Task Manager: Pydantic-schema-constrained decomposition + repair loop + static DAG validation (acyclicity, IO types, satisfiability). | Decomposition succeeds on ≥90% of the 20-task smoke suite within 2 repair attempts; every invalid DAG is rejected with a named violation. |
| W4 | Sequential executor with greedy exact-match resource binding; full span instrumentation; smoke suite + GAIA-subset baseline run. | **M1-EXIT review.** |

### Milestone M1-EXIT (hard gate)

- End-to-end execution of a multimodal task (e.g., "transcribe this audio file, summarize it, and cross-check the named figures against the database") touching ≥3 resource classes.
- ≥50% success on the 20-task smoke suite across 3 domains (research QA, document intelligence, code).
- Every kernel decision visible in Jaeger traces; Grafana dashboard shows per-resource cost/latency/utilization.
- Baseline numbers on GAIA subset recorded and tagged `baseline-v1`. **These are the control-group numbers for Papers 1–3.**

### Month-1 tests

- Unit: resource adapters, registry, DAG validator (property-based tests for acyclicity/type checks).
- Integration: full pipeline on smoke suite, nightly in CI against local models via replay cache.
- Chaos smoke (early): kill one Ollama container mid-run; confirm the failure is *visible* in traces (recovery comes in Month 3 — Month 1 only has to see it).

---

## MONTH 2 — Capability-Aware Orchestration (Modules M4–M7)

### Week-by-week

| Week | Build | Validate |
|---|---|---|
| W5 | Capability DNA schema + extractor (few-shot, constrained, confidence-gated escalation). Label 100 subtasks by hand for the agreement study. | Extractor vs human labels: ≥85% capability-flag agreement, ordinal scores within ±1 on ≥80%. |
| W6 | Feasibility filter + rule-based Pareto scorer; parallel dispatch of independent DAG branches; admission control (EDF-style SLO test, degrade-or-reject). | **Offline first:** router evaluated on RouterBench traces before any online spend. Online: router ≥ baseline quality on smoke suite. Admission control correctly rejects 10 synthetic infeasible tasks. |
| W7 | Shared Cognitive Memory: semantic cache with (DNA, embedding) keys; episodic store writing Experience Records; Redis execution blackboard with snapshot API. | Cache hit/stale-hit measured on a 50-task stream with 30% semantic overlap; snapshots restore a workflow mid-node in a scripted test. |
| W8 | Hierarchical agents on Ray (Supervisor/Sub-Supervisor/Worker), kernel-mediated NATS messaging, dynamic spawning + lifecycle reaping. Full Month-2 measured run. | **M2-EXIT review.** |

### Milestone M2-EXIT (hard gate)

- **H1 evidence (Paper 1):** router achieves equal-or-better quality than always-strongest-model at ≥30% lower cost on the GAIA subset; full Pareto curve plotted against always-cheapest, random-feasible, and a RouteLLM-style binary router.
- Parallel dispatch: ≥1.5× wall-clock speedup on the parallelizable subset of tasks.
- Semantic cache: ≥25% hit rate at ≤5% stale-hit rate on the 30%-overlap stream.
- Scheduler decision overhead <50 ms/node; kernel mediation overhead <5% of task latency.
- DNA extractor agreement study written up (feeds Paper 1's method section).

### Month-2 tests

- Offline router replay on RouterBench (free, reproducible) — gate before online runs.
- A/B: DNA-based matching vs Month-1 string matching (`--no-dna` ablation) on identical seeds.
- Load: 10 concurrent workflows; verify priority ordering and no blackboard cross-talk.

---

## MONTH 3 — Fault Tolerance, Learning, Full Evaluation (Modules M8–M10)

### Week-by-week

| Week | Build | Validate |
|---|---|---|
| W9 | Fault-injection harness: 8 injector classes (resource outage, silent model downgrade, corrupted tool output, delayed/dropped/mutated messages, poisoned memory, budget starvation), config-driven, seeded. | Each injector demonstrably fires and is visible in traces; injection rate calibration runs at 10/25/50%. |
| W10 | Failure Manager: detection ensemble → MAST-extended classifier → hand-crafted recovery table. Hot-swap with state handoff. | Detection precision/recall per fault class on injected runs; handoff success rate on 30 scripted mid-node failures. |
| W11 | Bandit-learned recovery selector; Learning Manager: experience writing, k-NN retrieval, subgraph transplantation, quality-prior updates, utility-based eviction. | Recovery bandit vs hand-crafted table on held-out fault mix; transplantation correctness spot-checks (20 manual reviews). |
| W12 | Full evaluation campaign: 200-task streams at 10/30/60% overlap; fault grid {0,10,25,50}% × {no-recovery, retry, Reflexion-style, table, bandit}; complete ablation matrix; data export + figures. | **M3-EXIT review.** |

### Milestone M3-EXIT (hard gate)

- **H2 evidence (Paper 2):** taxonomy-driven recovery beats naive retry by ≥20 percentage points task-completion under 25% fault injection; detection precision/recall and classification confusion matrix reported per fault class; hot-swap handoff success ≥70%.
- **H3 evidence (Paper 3):** marginal cost per task decreases ≥15% between first and last quartile of the 200-task stream at 30% overlap; reuse quality delta within 2 pp of from-scratch execution; stale-hit ≤5%.
- Complete ablation matrix (−DNA, −router, −memory, −recovery, −learning) on the full suite, seeded and replayable.
- All datasets, configs, injector definitions, and figures exported; artifact repo tagged `eval-v1`.

### Month-3 tests

- Determinism audit: two runs, same seeds, identical decision logs.
- Cost audit: total cloud spend for the final campaign within budget envelope (record/replay covers reruns).
- Third-party smoke: one person not on the project runs the artifact from README alone.

---

## Milestone Summary

| Gate | Date (relative) | Headline criterion |
|---|---|---|
| M1-EXIT | End W4 | Multimodal end-to-end run; ≥50% smoke success; baseline tagged |
| M2-EXIT | End W8 | Pareto dominance: equal quality, ≥30% cheaper; ≥1.5× parallel speedup; ≥25% cache hits |
| M3-EXIT | End W12 | +20 pp completion under 25% faults vs retry; ≥15% amortized cost drop; full ablations |

---

## Risk Register & Pivots

| # | Risk | L | I | Mitigation | Pivot if mitigation fails |
|---|---|---|---|---|---|
| R1 | LLM decomposition too unreliable to build on | High | High | Schema-constrained output, static validation, 2-attempt repair loop | Curated task-template library per domain; papers claim "validated decomposition," not "fully autonomous planning" |
| R2 | Routing gains marginal because model pool too homogeneous | Med | High | Force heterogeneity: weak-cheap local ↔ strong-expensive cloud; verify offline on RouterBench first | Re-center Paper 1 on **multimodal** routing (no SOTA exists) rather than text cost routing |
| R3 | Failure-detection precision too low → recovery fires on false positives, costs explode | Med | High | Per-detector confidence thresholds tuned on held-out injected faults; ensemble voting | Publish the detection characterization study itself ("how detectable are cognitive failures?") — a rigorous negative result is a workshop paper |
| R4 | Evaluation API costs explode | High | Med | Local-first development; replay cache; cloud only in final campaign | Shrink benchmark subsets; report local-model results as primary, cloud as spot-check |
| R5 | Scope creep (robotics, security, speech, enterprise) | Certain | Med | Non-goals fixed in spec: sensors/robots mocked, security = permission stub, one ASR resource | Cut mocked modalities from the *evaluation* but keep them in the *abstraction* (registry still proves generality) |
| R6 | Ray/NATS operational complexity eats research time | Med | Med | asyncio + Redis Streams remain the fallback path — kernel API identical | Ship single-process; distribution becomes "future work" without harming any paper claim |
| R7 | Bandit doesn't converge in available sample budget | Med | Med | Warm-start from rule-based scorer's decisions; restrict arm space via feasibility filter | Report rule-based Pareto scheduler as the contribution; bandit becomes an "online refinement" subsection |
| R8 | GAIA/τ-bench harness integration takes longer than planned | Med | Low | Use published community harnesses; subset to 50–100 tasks per benchmark | Custom 60-task in-house suite across 3 domains, fully released (helps the artifact anyway) |

---

## Resourcing & Budget Notes

- **Team assumption:** 1–2 implementers + 1 advisor. Ray/NATS (W8) and the evaluation campaign (W12) are the two crunch points — pre-book advisor review there.
- **Compute:** one GPU workstation (or modest cloud GPU) for Ollama/vLLM; cloud LLM budget reserved almost entirely for W6 (router online validation) and W12 (final campaign).
- **Data:** the 100-subtask human-labeled DNA set (W5) and the 20-manual-review transplantation audit (W11) are the only human-labeling costs — schedule them early in their weeks.
