"""Builds the default Capability Registry.

Every resource publishes a CapabilityManifest (DOC1 5.1). The numbers here are
hand-calibrated priors, not measurements -- DOC1 M1 is explicit that
quality_priors start as static config and are later overwritten by the Learning
Manager's observed validator statistics.

Resource ids deliberately match the coarse v0.0.2 capability strings
("web_search", "summarization", "vision") so a single register() call serves
both the DNA binding path and the exact-match fallback.
"""

from capability_registry import (
    Availability,
    CapabilityManifest,
    CapabilityRegistry,
    CostModel,
    IOSchema,
    LatencyModel,
)


def _print_manifest_table(manifests) -> None:
    print("\n[resource-registration] registered resources (DOC1 5.1 manifests)")
    print("=" * 72)
    for m in manifests:
        print(f"\n'{m.resource_id}'  class={m.resource_class}  risk={m.risk_class}")
        print(f"  provides   {', '.join(m.capabilities)}")
        print(
            f"  cost       ${m.cost_model.estimate_usd:.4f} "
            f"({m.cost_model.unit})"
        )
        print(
            f"  latency    p50={m.latency_model.p50_ms}ms "
            f"p95={m.latency_model.p95_ms}ms"
        )
        for flag, prior in sorted(m.quality_priors.items()):
            print(f"    prior    {flag:<26}{prior:.2f}")
    print()


def build_default_registry() -> CapabilityRegistry:
    """Register the three v0.0.2 capabilities plus a cheap summarizer variant.

    quick_summarization exists to give the Pareto scorer something to actually
    decide between: it overlaps summarization on text.summarization but is
    cheaper, faster and lower-quality there, while scoring *higher* on
    text.classification (skimming for key points is its whole job) and lacking
    reasoning.deep entirely. So a loose-budget summarize node picks the full
    summarizer, a tight-budget one picks this, and a compare/analyse node
    requiring reasoning.deep cannot pick it at all.
    """
    from capabilities import summarization, synthesis, vision, web_search

    manifests = [
        # Retrieval tool: DDG search plus an LLM pass that pulls exact facts out
        # of the results. Cheap per call, but the network hop dominates latency.
        CapabilityManifest(
            resource_id="web_search",
            resource_class="tool",
            capabilities=["web.search", "tool.calling", "reasoning.shallow"],
            input_schema=IOSchema(type="text", format="query"),
            output_schema=IOSchema(type="text", format="plain"),
            cost_model=CostModel(unit="per_call", estimate_usd=0.004),
            latency_model=LatencyModel(p50_ms=3000, p95_ms=9000),
            quality_priors={
                "web.search": 0.85,
                "tool.calling": 0.80,
                "reasoning.shallow": 0.60,
            },
            availability=Availability(status="up", rate_limit_rpm=30),
            # Pulls unvetted third-party web content into the workflow.
            risk_class="medium",
            metadata={
                "provider": "groq",
                "model": "qwen/qwen3.6-27b",
                "pipeline": "ddgs_search+groq",
            },
        ),
        # Full LLM pass. The only resource providing reasoning.deep, so every
        # compare/analyse node lands here regardless of the cost term.
        CapabilityManifest(
            resource_id="summarization",
            resource_class="llm",
            capabilities=[
                "text.summarization",
                "reasoning.deep",
                "reasoning.shallow",
                "planning.decomposition",
                "text.classification",
            ],
            input_schema=IOSchema(type="text", format="plain"),
            output_schema=IOSchema(type="text", format="plain"),
            cost_model=CostModel(unit="per_call", estimate_usd=0.003),
            latency_model=LatencyModel(p50_ms=1500, p95_ms=5000),
            quality_priors={
                "text.summarization": 0.85,
                "reasoning.deep": 0.80,
                "reasoning.shallow": 0.75,
                "planning.decomposition": 0.60,
                "text.classification": 0.55,
            },
            availability=Availability(status="up", rate_limit_rpm=60),
            risk_class="low",
            metadata={
                "provider": "groq",
                "model": "qwen/qwen3.6-27b",
            },
        ),
        # Local Ollama vision model: free per call, but slow, hence the wide p95.
        CapabilityManifest(
            resource_id="vision",
            resource_class="vlm",
            capabilities=[
                "vision.understanding",
                "reasoning.shallow",
                "text.classification",
            ],
            input_schema=IOSchema(type="image", format="path"),
            output_schema=IOSchema(type="text", format="plain"),
            cost_model=CostModel(unit="per_call", estimate_usd=0.0),
            latency_model=LatencyModel(p50_ms=6000, p95_ms=15000),
            quality_priors={
                "vision.understanding": 0.88,
                "reasoning.shallow": 0.60,
                "text.classification": 0.55,
            },
            availability=Availability(status="up", rate_limit_rpm=120),
            risk_class="low",
            metadata={
                "provider": "ollama",
                "model": "medgemma1.5:latest",
            },
        ),
        # Final-answer writer. Shares the model with `summarization` but not the
        # instruction: it expands rather than compresses, so its priors on
        # reasoning.deep beat the summarizer's and it is the natural winner for
        # the DAG's terminal node. Costlier per call because it runs at a much
        # larger max_tokens.
        CapabilityManifest(
            resource_id="synthesis",
            resource_class="llm",
            capabilities=[
                "answer.synthesis",
                "reasoning.deep",
                "text.summarization",
                "planning.decomposition",
                "reasoning.shallow",
            ],
            input_schema=IOSchema(type="text", format="plain"),
            output_schema=IOSchema(type="text", format="markdown"),
            cost_model=CostModel(unit="per_call", estimate_usd=0.006),
            latency_model=LatencyModel(p50_ms=3500, p95_ms=9000),
            quality_priors={
                # The only provider of answer.synthesis, so this prior sets the
                # bar the terminal node's min_quality has to clear.
                "answer.synthesis": 0.92,
                "reasoning.deep": 0.90,
                "text.summarization": 0.70,
                "planning.decomposition": 0.75,
                "reasoning.shallow": 0.70,
            },
            availability=Availability(status="up", rate_limit_rpm=60),
            risk_class="low",
            metadata={
                "provider": "groq",
                "model": "qwen/qwen3.6-27b",
            },
        ),
        # Cheap extractive pass over the same model with a terser instruction.
        CapabilityManifest(
            resource_id="quick_summarization",
            resource_class="llm",
            capabilities=[
                "text.summarization",
                "text.classification",
                "reasoning.shallow",
            ],
            input_schema=IOSchema(type="text", format="plain"),
            output_schema=IOSchema(type="text", format="plain"),
            cost_model=CostModel(unit="per_call", estimate_usd=0.001),
            latency_model=LatencyModel(p50_ms=700, p95_ms=2200),
            quality_priors={
                "text.summarization": 0.65,
                "text.classification": 0.80,
                "reasoning.shallow": 0.70,
            },
            availability=Availability(status="up", rate_limit_rpm=60),
            risk_class="low",
            metadata={
                "provider": "groq",
                "model": "qwen/qwen3.6-27b",
            },
        ),
    ]

    def quick_summarization_run(text, instruction=None):
        quick_instruction = instruction or (
            "Be concise: return only the essential key points, no elaboration."
        )
        return summarization.run(text, instruction=quick_instruction)

    run_fns = {
        "web_search": web_search.run,
        "summarization": summarization.run,
        "vision": vision.run,
        "synthesis": synthesis.run,
        "quick_summarization": quick_summarization_run,
    }

    registry = CapabilityRegistry()
    for manifest in manifests:
        registry.register(manifest, run_fns[manifest.resource_id])

    _print_manifest_table(manifests)
    print(
        f"[resource-registration] {len(manifests)} resources provide "
        f"{len(registry.provided_flags())} distinct capability flags: "
        f"{registry.provided_flags()}\n"
    )
    return registry


def build_hf_enabled_registry(
    provider: str | None = None,
) -> CapabilityRegistry:
    """Default pool plus the Hugging Face catalog resources.

    Opt-in on purpose: `build_default_registry()` stays the DOC2 baseline
    control group, so routing experiments can compare pools without touching
    the frozen default. The HF adapter is imported lazily so the core never
    depends on Hugging Face SDKs.
    """
    registry = build_default_registry()
    from providers.hf import register_hf_resources

    register_hf_resources(registry, provider=provider)

    # Print the FULL registry summary after all resources (default + HF) are
    # registered, so the summary accurately reflects what routing sees.
    all_manifests = list(registry.manifests())
    print("\n[resource-registration] full registry after HF resources added")
    print("=" * 72)
    _print_manifest_table(all_manifests)
    print(
        f"[resource-registration] {len(all_manifests)} resources provide "
        f"{len(registry.provided_flags())} distinct capability flags\n"
    )

    return registry
