# Oracle-Guided Self-Improving Meta-Systems — Reference Index

Context: Papers relevant to building deterministic, oracle-guided, explainable,
transparent self-improving systems for regulated domains (credit underwriting,
insurance approvals, etc.)

## Premise

The LLM is a fixed oracle. Everything else — prompts, strategy, control logic,
feedback interpretation, search heuristics — is the self-modifying program.
The output is deterministic, auditable code, not neural network predictions.

---

## TIER 1: Core Gödel Machine Lineage
*Systems that self-modify their own solver architecture*

### 01 — Huxley-Gödel Machine (HGM)
- **Authors**: Wenyi Wang, Piotr Piękos, et al. (incl. Schmidhuber)
- **Venue**: ICLR 2026 (Oral)
- **arXiv**: https://arxiv.org/abs/2510.21614
- **Code**: https://github.com/metauto-ai/HGM
- **Key idea**: Self-rewriting coding agent. Clade Metaproductivity (CMP) metric
  measures lineage fitness, not individual fitness. Proves CMP oracle is sufficient
  to implement a Gödel machine. Optimizes for evolvability, not immediate performance.
- **Relevance**: Formal foundation for self-improving decision systems. The clade
  insight (optimize descendants, not current instance) applies directly to
  evolving underwriting rule sets over time.

### 02 — Gödel Agent
- **Authors**: Xunjian Yin, Xinyi Wang, Liangming Pan, et al.
- **Venue**: ACL 2025
- **arXiv**: https://arxiv.org/abs/2410.04444
- **Key idea**: Agent recursively modifies its own reasoning loop via LLM prompting.
  Searches entire agent design space. No fixed optimization algorithm.
- **Relevance**: Framework for agents that improve their own decision-making
  methodology, not just their outputs.

### 03 — Darwin Gödel Machine (DGM)
- **Authors**: Sakana AI
- **Website**: https://sakana.ai/dgm/
- **Key idea**: Production implementation of self-modifying coding agent.
  Reads and rewrites its own Python codebase. Part of AI Scientist ecosystem.
- **Relevance**: Most production-ready self-modifying agent architecture.

---

## TIER 2: Program Evolution with LLM as Mutation Operator
*Systems that evolve deterministic programs using LLMs for variation*

### 04 — FunSearch
- **Authors**: DeepMind (Romera-Paredes, Barekatain, et al.)
- **Venue**: Nature (2024)
- **arXiv**: https://arxiv.org/abs/2402.12246
- **Key idea**: LLM generates candidate programs, island-based genetic algorithm
  evolves them, automated evaluator scores. Discovered new mathematics (cap-set).
  Program skeleton fixed, critical logic evolved.
- **Relevance**: **Most directly applicable to regulated domains.** Produces
  deterministic programs. Evaluation-driven. Could evolve underwriting rules,
  pricing functions, approval logic the same way it evolves bin-packing algorithms.

### 05 — CodeEvolve
- **Authors**: Souza et al. (Inter & Co)
- **arXiv**: https://arxiv.org/abs/2510.14150
- **Code**: https://github.com/inter-co/science-codeevolve
- **Key idea**: Open-source FunSearch/AlphaEvolve competitor. Island-based GA +
  LLM mutations + adaptive meta-prompting. Shows open-weight models match
  closed-source baselines.
- **Relevance**: Open-source starting point for building custom program evolution
  systems. Directly forkable.

### (Also: AlphaEvolve — DeepMind, 2025. Generalizes FunSearch to entire codebases. Closed source.)

---

## TIER 3: Prompt/Strategy Self-Modification
*Systems that evolve how they use the oracle*

### 06 — PromptBreeder
- **Authors**: Fernando et al. (Google DeepMind)
- **Venue**: ICML 2024
- **arXiv**: https://arxiv.org/abs/2309.16797
- **Key idea**: Second-order self-improvement. Evolves task-prompts AND the
  meta-prompts that guide prompt mutation. The mutation strategy itself evolves.
- **Relevance**: Shows that the "how to query the oracle" layer is itself
  evolvable. Applicable to evolving prompt strategies for decision systems.

### 07 — DSPy
- **Authors**: Khattab et al. (Stanford NLP)
- **arXiv**: https://arxiv.org/abs/2310.03714
- **Code**: https://github.com/stanfordnlp/dspy
- **Key idea**: Framework for programming LM pipelines. Compiles high-level
  signatures into optimized prompts via Bayesian optimization. Treats scaffolding
  as a program to be compiled.
- **Relevance**: Practical framework for building optimizable LLM pipelines.
  Could serve as the scaffolding layer for a regulated decision system.

---

## TIER 4: Iterative Refinement & Feedback Systems
*Systems that improve outputs through structured feedback loops*

### 08 — Reflexion
- **Authors**: Shinn et al.
- **arXiv**: https://arxiv.org/abs/2303.11366
- **Key idea**: Verbal reinforcement learning. Agent reflects on failures,
  stores reflections in episodic memory, conditions future decisions on memory.
- **Relevance**: Memory-augmented refinement. Applicable to systems that need
  to learn from past decision errors (e.g., claim denials that were overturned).

### 09 — Self-Refine
- **Authors**: Madaan et al.
- **Venue**: ICLR 2023
- **arXiv**: https://arxiv.org/abs/2303.17651
- **Website**: https://selfrefine.info/
- **Key idea**: Same model generates, critiques, and refines. ~20% improvement
  over single-pass. Alternating FEEDBACK and REFINE steps.
- **Relevance**: Simplest instantiation of the refinement loop. Baseline
  architecture for iterative decision improvement.

### 10 — LATS (Language Agent Tree Search)
- **Authors**: Zhou et al.
- **Venue**: ICML 2024
- **arXiv**: https://arxiv.org/abs/2310.04406
- **Key idea**: Monte Carlo Tree Search + LLM value functions + environment
  feedback + self-reflection. Unifies reasoning, acting, planning.
- **Relevance**: Structured search over decision trees with backtracking.
  Natural fit for multi-factor approval decisions.

### 11 — RISE (Recursive Introspection)
- **Authors**: Qu et al.
- **Venue**: NeurIPS 2024
- **arXiv**: https://arxiv.org/abs/2407.18219
- **Key idea**: Teaches LLMs self-improvement through multi-turn MDPs.
  Bootstraps on-policy rollouts with best-of-N revision candidates.
- **Relevance**: Shows self-improvement capability can be taught, not just
  engineered. Could train domain-specific self-improving reasoners.

---

## FOUNDATIONAL

### 12 — Gödel Machines: Self-Referential Universal Problem Solvers
- **Author**: Jürgen Schmidhuber (2003)
- **arXiv**: https://arxiv.org/abs/cs/0309048
- **Key idea**: The original. A general task solver that optimally makes any
  provable self-improvements. Runs a proof searcher looking for formal proofs
  that code modifications yield higher expected utility. Theoretically optimal
  but practically intractable — all subsequent work approximates this.

---

## RELATED SYSTEM (not a paper)

### Poetiq — ARC-AGI-2 solver (unrelated to this project)
- **Code**: https://github.com/poetiq-ai/poetiq-arc-agi-solver
- **Blog**: https://poetiq.ai/posts/arcagi_verified/
- **Relevance**: an LLM-agnostic "converge to code" refinement loop on ARC-AGI-2;
  see their write-up for reported results.

---

## Reading Order (Suggested)

1. **Schmidhuber 2003** (#12) — Foundational theory
2. **Huxley-Gödel Machine** (#01) — Modern practical approximation
3. **FunSearch** (#04) — Program evolution with deterministic outputs
4. **Self-Refine** (#09) — Simplest refinement loop
5. **Poetiq blog** — a refinement loop applied at scale
6. Everything else based on interest

---

## For the Regulated Domain Application

**Start with**: FunSearch (#04) + CodeEvolve (#05) for the program synthesis backbone.
**Add**: HGM (#01) for the self-improvement framework.
**Layer**: DSPy (#07) for the scaffolding optimization.
**Ground**: Reflexion (#08) for learning from past decision errors.

The output at every stage should be deterministic code — auditable, versionable,
explainable to regulators. The oracle (LLM) is used at development time only.
