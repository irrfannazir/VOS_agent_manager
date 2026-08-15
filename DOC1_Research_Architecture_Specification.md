# DOCUMENT 1 — Formal Research & Architecture Specification

**Project:** AOS — Adaptive Cognitive AI Microkernel for Domain-Independent Multi-Agent Orchestration
**Document type:** Research & Architecture Specification (v1.0)
**Date:** July 2026

---

## 1. Abstract

Contemporary multi-agent AI frameworks (AutoGen/AG2, LangGraph, CrewAI, MetaGPT, CAMEL) provide orchestration primitives — role definitions, message passing, graph-based control flow — but delegate the hard systems problems to hand-written application logic: which resource executes which subtask, what happens when a model hallucinates or a tool fails, and how knowledge from one execution transfers to the next. Recent "agent operating system" proposals, most notably AIOS (Mei et al., 2024), introduce kernel-level scheduling for LLM agents, yet remain LLM-centric, employ capability-blind FIFO/round-robin scheduling, and lack principled failure recovery or cross-task learning.

This specification defines **AOS**, an Adaptive Cognitive AI Microkernel that elevates three mechanisms to kernel level:

1. **Capability DNA** — a typed, machine-readable capability vector that unifies LLMs, vision models, speech models, tools, APIs, databases, and (simulated) sensor/robotic interfaces under a single schedulable resource abstraction, enabling feasibility analysis and Pareto-optimal cost/quality/latency scheduling.
2. **Cognitive Fault Tolerance** — a closed-loop pipeline that detects, classifies (extending the MAST failure taxonomy, Berkeley 2025), and recovers from planning, reasoning, resource, tool, communication, and memory failures using a learned recovery-policy selector, including mid-workflow resource hot-swapping with state handoff.
3. **Experience-Driven Workflow Optimization** — an episodic memory of executed workflow graphs enabling semantic subgraph reuse, cached-result transplantation, and amortized cost reduction across sustained task streams.

We hypothesize that (H1) capability-aware scheduling Pareto-dominates single-best-model execution on cost–quality frontiers, including multimodal tasks where no routing SOTA exists; (H2) taxonomy-driven recovery outperforms retry and reflection baselines under controlled fault injection; and (H3) workflow-graph experience reuse yields monotonically decreasing marginal cost over task streams with semantic overlap. The architecture is validated on public agentic benchmarks (GAIA, τ-bench, SWE-bench-Lite subsets) and a purpose-built multimodal fault-injection suite released as an open research artifact.

---

## 2. State of the Art and Gap Analysis

### 2.1 Related work map

| Area | Representative SOTA | What it does | What it does NOT do |
|---|---|---|---|
| Agent orchestration frameworks | AutoGen/AG2, LangGraph, CrewAI, MetaGPT, CAMEL | Role-based agents, graph control flow, conversation patterns | No kernel; routing, recovery, memory are application code; topologies largely static |
| Agent operating systems | AIOS (Mei et al. 2024), AIOS-Agent SDK | Kernel with agent scheduler, context manager, tool manager | LLM-centric; FIFO/RR scheduling; no capability typing; no failure taxonomy; no cross-task learning |
| LLM routing | RouteLLM, FrugalGPT, Hybrid-LLM, RouterBench, CASTER | Cost/quality routing between 2–N LLMs on text tasks | Text-only; pairwise or cascade; no tools/sensors/vision in the decision space; no per-subtask DAG routing |
| Multi-agent failure analysis | MAST taxonomy (2025), Reflexion, self-refine | Categorizes failures; retry with self-critique | Taxonomy is descriptive only — no automated detection→classification→recovery loop |
| Agent memory | MemGPT/Letta, Mem0, A-MEM, GPTCache | Long-term conversational memory; response-level semantic caching | Per-agent, text-level; no kernel-level workflow-graph storage; no subgraph reuse |
| Experience learning | ExpeL, Voyager skill library, AgentTuning, DSPy/GEPA | Accumulate textual insights/skills; prompt optimization | Per-agent skills, not transferable workflow structures; no resource-binding reuse |
| Interoperability | MCP (Anthropic), A2A (Google) | Standard tool and agent-to-agent protocols | Protocols only — no scheduling or capability semantics on top |

### 2.2 Identified gaps (the research thesis)

- **G1 — No typed capability abstraction.** No existing system represents heterogeneous resources (models + tools + APIs + sensors) in one typed, schedulable formalism. Routing literature optimizes "which LLM"; AOS generalizes to "which capability bundle."
- **G2 — Modality-blind, per-task routing.** SOTA routers make one decision per user query on text. AOS routes per DAG node, across modalities, under joint cost/latency/quality/risk constraints.
- **G3 — Open-loop failure handling.** MAST proves multi-agent systems fail in taxonomizable ways; production systems still only retry. The detection→classification→learned-recovery loop is unclaimed territory.
- **G4 — Response-level memory only.** Caching SOTA (GPTCache) stores single responses. Storing and *transplanting validated workflow subgraphs* (structure + resource bindings + intermediate outputs) is novel.
- **G5 — No admission control / schedulability notion.** Real-time OS theory (EDF, admission tests) has never been mapped onto cognitive task execution with latency SLOs and cost ceilings.

---

## 3. Design Principles

1. **Everything is a Resource.** Models, tools, APIs, databases, sensors, robots — one contract, one registry.
2. **The kernel mediates everything.** No peer-to-peer agent communication; all messages traverse the kernel bus. Mediation is what makes observation, failure detection, and recovery possible.
3. **Capabilities, not models.** Scheduling decisions are made in capability space; concrete resources are bound late.
4. **Fail cognitively, recover deliberately.** Failures are classified before recovery is selected; recovery is a policy, not a reflex.
5. **Every execution teaches the kernel.** Workflow graphs, outcomes, and costs are first-class stored artifacts.
6. **Incremental delivery.** Baseline → Integration → Novelty. Each phase produces a runnable, measurable system.
7. **Telemetry from day one.** Every kernel decision is an OpenTelemetry span attribute; paper metrics fall out of traces.

---

## 4. Technology Stack Analysis

| Layer | Choice | Justification for experimental research |
|---|---|---|
| Kernel runtime | Python 3.12 + asyncio (Phase A); Ray actors (Phase B+) | asyncio keeps the microkernel single-process, deterministic, and debuggable while the abstractions stabilize. Ray adds actor lifecycle, placement groups, and fault domains for distributed agents without changing kernel APIs. |
| Model gateway | LiteLLM; vLLM + Ollama for local models | LiteLLM normalizes 100+ providers behind one call signature — it *is* the Unified Resource Interface for the model class. Local models are mandatory for research: (a) reproducible costs, (b) genuine heterogeneity (weak-cheap vs strong-expensive) so routing experiments have signal, (c) fault injection without burning API budget. |
| Tool ABI | Model Context Protocol (MCP) | De-facto standard tool interface. Wrapping every tool/API/mock-sensor as an MCP server means the Capability Registry ingests standard manifests, and the paper gains an interoperability claim. |
| Messaging | Redis Streams (Phase A) → NATS (Phase B) | Kernel-observed subjects per agent; the Communication Manager sees all traffic — prerequisite for communication-failure detection and for the message-corruption fault injector. |
| Vector memory | Qdrant (or pgvector) + one embedding model | Backs semantic cache and episodic retrieval. Cache keys are (Capability DNA, task embedding) pairs — a novel key design vs GPTCache's response-text keys. |
| Episodic store | PostgreSQL, workflow graphs as JSONB | Graphs must be queryable post-hoc (Learning Manager, ablation analysis, paper figures). JSONB + GIN indexes suffice at research scale. |
| Live execution state | Redis (hash/blackboard per workflow) | Shared execution memory readable by all agents; snapshot source for hot-swap state handoff. |
| Observability | OpenTelemetry → Prometheus + Grafana; Jaeger for traces | Route chosen, DNA vector, cache hit/miss, recovery fired — all span attributes. Monitoring Manager becomes mostly configuration. |
| Routing/recovery policy learning | Contextual bandits: LinUCB / Thompson sampling (Vowpal Wabbit or ~200-line custom) | Context = Capability DNA; arms = resource assignments (or recovery strategies); reward = quality − λ·cost − μ·latency. Bandits are sample-efficient and publishable in 3 months; full RL is not. |
| Structured planning | Pydantic-schema-constrained LLM decomposition; DSPy optional for prompt optimization | Free-form planning is the top reliability risk. JSON-schema-forced task graphs get static validation (cycles, types, capability references) before any execution. |
| Benchmarks | GAIA, τ-bench, SWE-bench-Lite (subsets); RouterBench traces for offline router eval; custom multimodal suite | Domain-independence claim requires ≥3 domains. RouterBench enables free offline router evaluation before paid online runs. |
| Quality estimation | Conformal prediction wrapper over router quality estimates | Converts heuristic routing into routing with statistical guarantees (quality ≥ threshold with prob ≥ 1−α). Reviewers reward guarantees. |

**Explicit non-goals (scope control):** real robot/sensor drivers (simulated as MCP capability endpoints), production-grade Security Manager (permission-check stub only), speech beyond one transcription resource, enterprise multi-tenancy.

---

## 5. Core Data Models

### 5.1 CapabilityManifest (published by every Resource)

```json
{
  "resource_id": "gpt-strong-cloud-01",
  "resource_class": "llm | vlm | asr | tts | tool | api | database | sensor | robot",
  "capabilities": ["reasoning.deep", "code.generation", "tool.calling"],
  "input_schema":  {"type": "text|image|audio|structured", "format": "..."},
  "output_schema": {"type": "text|image|audio|structured", "format": "..."},
  "cost_model":   {"unit": "per_1k_tokens|per_call", "estimate_usd": 0.01},
  "latency_model":{"p50_ms": 1200, "p95_ms": 4000},
  "quality_priors": {"reasoning.deep": 0.86, "code.generation": 0.78},
  "availability": {"status": "up", "rate_limit_rpm": 60},
  "risk_class": "low | medium | high"
}
```

### 5.2 Capability DNA (per subtask, produced by the Task Manager)

Fixed-dimension typed vector with three segments:

- **Discrete capability flags** — required capability identifiers (vision.understanding, speech.transcription, reasoning.deep, tool.calling, db.query, code.generation, …).
- **Ordinal complexity scores (0–4)** — reasoning depth, planning horizon, tool-orchestration complexity, memory dependence, parallelizability.
- **Continuous constraints** — cost ceiling (USD), latency SLO (ms), minimum quality threshold (0–1), risk tolerance.

```json
{
  "flags": ["vision.understanding", "reasoning.deep"],
  "ordinals": {"reasoning_depth": 3, "planning_horizon": 1, "tool_complexity": 0,
               "memory_dependence": 2, "parallelizability": 0},
  "constraints": {"cost_ceiling_usd": 0.05, "latency_slo_ms": 8000,
                  "min_quality": 0.7, "risk_tolerance": "medium"}
}
```

### 5.3 Task DAG

Nodes = subtasks `{id, description, dna, input_bindings, status}`; edges = data dependencies. Validated statically: acyclicity, type compatibility between bound outputs and inputs, every DNA flag satisfiable by ≥1 registered resource (else admission control rejects or degrades).

### 5.4 Experience Record (written by Learning Manager on task completion)

```json
{
  "task_embedding": "[...]",
  "dna_profile": "aggregate DNA of the DAG",
  "workflow_graph": "full DAG with resource bindings and per-node outcomes",
  "metrics": {"success": true, "cost_usd": 0.42, "latency_ms": 31200,
              "recoveries_fired": 1, "cache_hits": 3},
  "failures": [{"class": "tool.output_corrupt", "recovery": "resource_substitution", "recovered": true}]
}
```

---

## 6. Incremental Module Architecture

### PHASE A — Baseline Microkernel (Weeks 1–4)

#### M1 — Unified Resource Interface & Capability Registry
Every resource implements exactly two operations: `describe() → CapabilityManifest` and `invoke(request) → response`. Adapters: LiteLLM wrapper (all models), MCP client wrapper (all tools/APIs), mock adapters for sensor/robot endpoints (return scripted, capability-consistent data — the abstraction is the contribution, not device drivers). The Registry maintains the live manifest table, watches availability via heartbeats, and exposes `find(capability_flags) → [resource_id]`. Implementation detail: manifests are re-published on change; quality_priors start as static config and are later overwritten by the Learning Manager's observed statistics — this is the loop that makes the registry "learned."

#### M2 — Task Manager (schema-constrained decomposition)
An LLM planner is forced through a Pydantic JSON schema to emit a Task DAG. A repair loop (max 2 attempts) feeds validation errors back to the planner. Static validation gate: acyclicity, IO type checks, capability satisfiability. If unsatisfiable → structured rejection with the missing capability named (this is the seed of admission control). Requirement extraction and dependency identification are part of the same constrained generation pass; DNA extraction is stubbed in Phase A (string capability tags only) and upgraded in M4.

#### M3 — Sequential Executor + Monitoring Manager
Walks the DAG in topological order; greedy resource matching by exact capability string. Every node execution emits an OpenTelemetry span with attributes: resource chosen, tokens, cost, latency, validation result. Prometheus counters: task success, per-resource utilization, cumulative spend. **This module is deliberately naive — it is the control group for every subsequent experiment. Freeze it once M1-EXIT passes.**

### PHASE B — Integration (Weeks 5–8)

#### M4 — Capability DNA Extractor
Dedicated extraction pass mapping each subtask description → full DNA vector (Section 5.2). Implementation: few-shot constrained generation with a small held-out human-labeled set (100 subtasks) for agreement measurement. The extractor is itself a routed resource (cheap model first, escalate on low confidence). DNA replaces string matching everywhere downstream.

#### M5 — Capability-Aware Scheduler + Dynamic Router
Two-stage decision per DAG node:
1. **Feasibility filter:** `resource.capabilities ⊇ dna.flags` AND constraint satisfiability (cost/latency/risk).
2. **Policy scoring:** Phase B ships a rule-based Pareto scorer `score = quality_prior − λ·cost_est − μ·latency_est`; Phase C swaps in the contextual bandit (context = DNA, arms = feasible resources, reward observed from validators). ε-greedy exploration bounded to low-risk nodes.

Scheduler also performs: parallel dispatch of independent DAG branches (asyncio gather / Ray tasks), priority ordering across concurrent workflows, and **admission control** — EDF-inspired: given declared latency SLOs and current queue, reject or degrade (relax min_quality, allow cheaper resources) infeasible tasks. This is "cognitive deadline scheduling" — no prior art.

#### M6 — Shared Cognitive Memory
Four stores behind one MemoryManager API:
- **Semantic cache** — key = (DNA vector, subtask embedding); hit if cosine ≥ τ AND DNA-compatible. Returns prior validated output. τ tuned to control stale-hit rate.
- **Episodic memory** — Experience Records (5.4) in Postgres, embeddings in Qdrant.
- **Execution memory** — Redis blackboard per workflow: intermediate results, agent scratch state. Snapshot-able (needed for hot-swap in M8).
- **Semantic memory** — long-lived facts/knowledge distilled by the Learning Manager.

#### M7 — Hierarchical Agent Manager + Communication Manager
Supervisor → Sub-Supervisor → Worker agents as Ray actors. Agents are *decision-makers only*; all computation goes through Resources. All messages traverse kernel-owned NATS subjects (`agent.{id}.inbox`), giving the kernel a total order of communication for observation and fault injection. Dynamic spawning: Supervisor requests agent instantiation from Agent Manager with a role spec; lifecycle (spawn, idle-reap, kill) is kernel-owned.

### PHASE C — Advanced Novelty (Weeks 9–12)

#### M8 — Cognitive Failure Manager
Closed loop, three stages:
1. **Detection ensemble:** schema validators on every resource output; LLM-judge quality scorer on node outputs; timeout/cost-overrun monitors; inter-agent contradiction checker (compares claims on the blackboard); memory-staleness checker.
2. **Classification:** symptoms → extended MAST taxonomy: {planning.bad_decomposition, reasoning.hallucination, reasoning.inconsistency, resource.outage, resource.degraded, tool.output_corrupt, communication.lost, communication.corrupt, memory.stale}.
3. **Recovery policy selector:** strategies = {retry_same, retry_with_feedback, resource_substitution, hot_swap_with_state_handoff, subtask_re-decomposition, agent_reassignment, memory_retrieval_answer, workflow_regeneration}. Ships as a hand-crafted taxonomy→strategy table; upgraded to a bandit (context = failure class + DNA + budget remaining; reward = recovery success − recovery cost).

**Hot-swap with state handoff:** on resource failure mid-node, the executor snapshots the node's execution-memory slice, selects a substitute via the feasibility filter, transplants the snapshot into the substitute's context, and resumes. Handoff success rate is a headline metric.

#### M9 — Learning Manager
On completion: write Experience Record; update resource quality_priors (exponential moving average of validator scores per capability). On new task: k-NN retrieval over episodic memory (task embedding + DNA profile); attempt **subgraph transplantation** — graft matching validated subgraphs (structure + resource bindings) into the new plan; substitute cached node outputs where semantic-cache hits occur. Utility-learned eviction: experiences whose reuse historically lowered quality get down-weighted and eventually evicted (memory reclamation, completing the OS analogy).

#### M10 — Evaluation & Fault-Injection Harness
Config-driven injectors, deterministic via seeds: resource outage, silent model downgrade, corrupted tool output, delayed/dropped/mutated messages, poisoned memory entries, budget starvation. Benchmark runners for GAIA/τ-bench/SWE-bench-Lite subsets + the custom multimodal suite. Ablation switches: `--no-dna`, `--no-router`, `--no-memory`, `--no-recovery`, `--no-learning`. Record/replay cache of all resource responses so re-runs are free and deterministic. **Released as an open artifact with the papers.**

---

## 7. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Scheduler decision overhead | < 50 ms per node (excluding DNA extraction) |
| Kernel mediation overhead | < 5% of end-to-end task latency |
| Reproducibility | All measured runs seeded + record/replay; one-command re-execution |
| Ablation coverage | Every novel module independently toggleable |
| Telemetry completeness | 100% of kernel decisions present as span attributes |
| Artifact quality | Fault-injection suite + configs public, documented, runnable by third parties |

---

## 8. Glossary

- **Resource** — any invokable computational component publishing a CapabilityManifest.
- **Capability DNA** — typed per-subtask requirement vector (flags + ordinals + constraints).
- **Feasibility filter** — set-containment + constraint check producing candidate resources.
- **Cognitive failure** — any deviation detected at the semantic level (not just exceptions).
- **Hot swap** — mid-node resource replacement with execution-state transplantation.
- **Subgraph transplantation** — reuse of a validated prior workflow fragment (structure + bindings + outputs) in a new plan.
- **Admission control** — pre-execution schedulability test against latency SLOs and cost ceilings.
