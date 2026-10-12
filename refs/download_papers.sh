#!/bin/bash
# Download script for oracle-guided self-improving meta-systems papers
# Run this on your local machine: bash download_papers.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Downloading papers to: $DIR"
echo ""

# === TIER 1: Core Gödel Machine Lineage ===

echo "[1/12] Huxley-Gödel Machine (KAUST/Schmidhuber, ICLR 2026 Oral)"
curl -sL -o "$DIR/01_Huxley_Godel_Machine.pdf" "https://arxiv.org/pdf/2510.21614"

echo "[2/12] Gödel Agent — Self-Referential Recursive Self-Improvement (ACL 2025)"
curl -sL -o "$DIR/02_Godel_Agent.pdf" "https://arxiv.org/pdf/2410.04444"

echo "[3/12] Darwin Gödel Machine (Sakana AI)"
curl -sL -o "$DIR/03_Darwin_Godel_Machine.pdf" "https://arxiv.org/pdf/2505.22535"

# === TIER 2: Program Evolution with LLM as Mutation Operator ===

echo "[4/12] FunSearch — Mathematical Discoveries via LLMs (DeepMind, Nature)"
curl -sL -o "$DIR/04_FunSearch.pdf" "https://arxiv.org/pdf/2402.12246"

echo "[5/12] CodeEvolve — Open-Source Evolutionary Coding Agent"
curl -sL -o "$DIR/05_CodeEvolve.pdf" "https://arxiv.org/pdf/2510.14150"

# === TIER 3: Prompt/Strategy Self-Modification ===

echo "[6/12] PromptBreeder — Self-Referential Prompt Evolution (ICML 2024)"
curl -sL -o "$DIR/06_PromptBreeder.pdf" "https://arxiv.org/pdf/2309.16797"

echo "[7/12] DSPy — Compiling LM Calls into Self-Improving Pipelines (Stanford)"
curl -sL -o "$DIR/07_DSPy.pdf" "https://arxiv.org/pdf/2310.03714"

# === TIER 4: Iterative Refinement & Feedback Systems ===

echo "[8/12] Reflexion — Verbal Reinforcement Learning"
curl -sL -o "$DIR/08_Reflexion.pdf" "https://arxiv.org/pdf/2303.11366"

echo "[9/12] Self-Refine — Iterative Refinement with Self-Feedback (ICLR 2023)"
curl -sL -o "$DIR/09_Self_Refine.pdf" "https://arxiv.org/pdf/2303.17651"

echo "[10/12] LATS — Language Agent Tree Search (ICML 2024)"
curl -sL -o "$DIR/10_LATS.pdf" "https://arxiv.org/pdf/2310.04406"

echo "[11/12] RISE — Recursive Introspection (NeurIPS 2024)"
curl -sL -o "$DIR/11_RISE_Recursive_Introspection.pdf" "https://arxiv.org/pdf/2407.18219"

# === FOUNDATIONAL ===

echo "[12/12] Original Gödel Machines (Schmidhuber 2003)"
curl -sL -o "$DIR/12_Godel_Machines_Schmidhuber_2003.pdf" "https://arxiv.org/pdf/cs/0309048"

echo ""
echo "Done. Downloaded 12 papers to $DIR"
ls -lh "$DIR"/*.pdf 2>/dev/null | awk '{print $5, $9}'
