<div align="center">

# LiteResearcher

### A Scalable Agentic RL Training Framework for Deep Research Agent

[Paper](#) · [Project Page](https://wanli-lee.github.io/LiteResearcher/) · [Model (coming soon)](#)

</div>

**LiteResearcher** is a training framework that makes Agentic RL scalable for deep research agents. By constructing a lite virtual world that mirrors real-world search dynamics, we enable a continuously improving training recipe that empowers a tiny 4B search agent to outperform large-scale open-source and commercial models.

**LiteResearcher-4B** achieves **71.3%** on GAIA and **78.0%** on Xbench-DeepSearch, surpassing models up to 8× larger (Tongyi DeepSearch 30B, WebSailor 30B) and matching commercial systems (Claude-4.5-Sonnet, GPT-5).

## Results

| Model | GAIA | BrowseComp | HLE | Frames | WebWalker | Seal-0 | Xbench-DS |
|-------|------|-----------|-----|--------|-----------|--------|-----------|
| GPT-5-high | 76.4% | 54.9% | 35.2% | — | — | 51.4% | 77.8% |
| Claude-4.5-Sonnet | 71.2% | 19.6% | 24.5% | 85.0% | — | 53.4% | 66.0% |
| Tongyi DeepSearch 30B | 70.9% | 43.4% | 32.9% | 90.6% | 72.2% | — | 75.0% |
| AgentCPM-Explore 4B | 63.9% | 24.1% | 19.1% | 82.7% | 68.1% | 40.5% | 70.0% |
| **LiteResearcher-4B** | **71.3%** | 27.5% | 22.0% | 83.1% | **72.7%** | **41.8%** | **78.0%** |

## Method Overview

<div align="center">
<img src="docs/static/overview.png" width="90%">
</div>

Three pillars enable scalable Agentic RL:

1. **Co-construct Training Data & Corpus** — Scale up information sources with a simple-but-effective synthesis pipeline, then co-evolve training QA pairs and the local webpage corpus.
2. **Stable Local Tool Environment** — Build local search engine (Milvus + BGE-M3) and local browse tool (PostgreSQL) from ~32M real webpages, achieving 10–46× speedup at zero marginal cost.
3. **Difficulty-Aware Curriculum RL** — Multi-stage curriculum with on-policy GRPO, filtering tasks by pass@8 difficulty to sustain monotonic improvement.

## Repository Structure

```
├── Eval/                   # Evaluation (released)
├── Training/               # RL training (coming soon)
├── DataGen/                # Data synthesis (coming soon)
├── Environment/            # Local search/browse environment (coming soon)
└── docs/                   # Project page
```

## Quick Start — Evaluation

```bash
cd Eval
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set MODEL, SERPER_KEY_ID, SCRAPEDO_API_KEY

# Start model server (SGLang/vLLM)
bash scripts/start_sglang.sh

# Run evaluation
bash scripts/run_all.sh
```

See [`Eval/README.md`](Eval/README.md) for detailed configuration and usage.

## Release Plan

- [x] Evaluation code
- [x] Project page
- [ ] Model weights (LiteResearcher-4B)
- [ ] Training code (GRPO + curriculum RL)
- [ ] Data synthesis pipeline
- [ ] Local search/browse environment setup

## Citation

```bibtex
@article{li2026literesearcher,
  title={LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent},
  author={Li, Wanli and Qu, Bince and Pan, Bo and Zhang, Jianyu and Liu, Zheng and Zhang, Pan and Chen, Wei and Zhang, Bo},
  year={2026}
}
```

## License

Apache 2.0
