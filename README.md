# MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification

Official implementation of the ICML 2026 paper
**"MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification"**.

**Remarks**: This codebase was cleaned and reorganized for release with
[Claude Code](https://claude.com/claude-code) — removing baseline-only code and unused
helper files, and restructuring the project into the `cv/` and `nlp/` pipelines.

MetaMoE unifies independently trained, domain-specialized experts into a single deployable
Mixture-of-Experts (MoE) model **without sharing private client data**, using public *proxy data*
as surrogates for the inaccessible private distributions. It has three stages:

1. **Proxy data selection** — a relevance-weighted Determinantal Point Process (DPP) selects
   proxy samples from a public dataset that are both *relevant* to a client domain and *diverse*.
2. **Proxy-aligned expert training** — each client finetunes the FFN sublayers of the seed model
   on its private data **together with** its proxy data.
3. **Context-aware router training** — experts are merged into a single MoE and a context-aware
   router is trained on the union of proxy datasets.

See `metamoe.pdf` for the full paper.

## Repository structure

```
.
├── cv/                       # Computer-vision pipeline (CLIP ViT)
│   ├── mm_create_proxy_data.py         # Stage 1: relevance-weighted DPP proxy selection
│   ├── mm_train.py                     # Stage 2: proxy-aligned expert training (LoRA on FFN)
│   ├── mm_compute_embedding.py         # router embedding from trained experts (Eq. 7)
│   ├── mm_eval_mlp.py                  # Stage 3: context-aware router training + evaluation
│   ├── mm_utils.py                     # shared CV helpers
│   └── scripts/                        # one runnable bash script per step
├── nlp/                      # Natural-language pipeline (LLaMA)
│   ├── nlp_create_proxy_data.py        # Stage 1: relevance-weighted DPP proxy selection
│   ├── nlp_train.py                    # Stage 2: proxy-aligned expert training (LoRA on FFN)
│   ├── nlp_compute_router_embedding.py # router embedding from trained experts (Eq. 7)
│   ├── nlp_eval_ours.py                # Stage 3: context-aware router training + evaluation
│   ├── data.py                         # dataset loading / formatting
│   ├── configs/                        # training / eval hyper-parameter configs
│   └── scripts/                        # one runnable bash script per step
├── src/                      # Shared library
│   ├── lora/                           # LoRA adapter implementation
│   ├── mlp_mole/                       # context-aware MoE-of-LoRA layers (the unified MoE)
│   ├── multi_modal/datasets/           # CV dataset loaders
│   └── utils/                          # DPP MAP inference, constants, shared helpers
├── requirements.txt
└── metamoe.pdf
```

## Installation

```bash
conda create -n metamoe python=3.10 -y
conda activate metamoe

pip install -r requirements.txt
# OpenAI CLIP (required for the CV pipeline)
pip install git+https://github.com/openai/CLIP.git
```

## Configuration

`src/utils/constants.py` ships with four placeholder values, each set to `"TODO"`.
**Before running anything, replace every `"TODO"` with the correct value for your environment:**

```python
ACCESS_TOKEN      = "TODO"   # your HuggingFace access token (needed to download gated models, e.g. LLaMA)
DEFAULT_CACHE_DIR = "TODO"   # HuggingFace cache directory for downloaded models and datasets
DEFAULT_DATA_DIR  = "TODO"   # root directory of the CV image datasets (see "Data preparation" below)
HOME_PATH         = "TODO"   # absolute path to this repository (used to locate outputs_train/)
```

- `ACCESS_TOKEN` — create a token at <https://huggingface.co/settings/tokens> and accept the
  LLaMA license; only needed for the NLP pipeline.
- `DEFAULT_CACHE_DIR` — any writable directory (e.g. `~/.cache/huggingface`).
- `DEFAULT_DATA_DIR` — the directory where you placed the CV datasets.
- `HOME_PATH` — the absolute path of this cloned repository.

All bash scripts in `cv/scripts/` and `nlp/scripts/` are meant to be launched from their own
directory; they `cd` to the repository root and set `PYTHONPATH=.` automatically.

## Data preparation

### NLP datasets — downloaded automatically from HuggingFace

The NLP datasets are pulled on first use via `datasets.load_dataset()`, no manual download needed:

| Dataset | HuggingFace ID | Role |
|---|---|---|
| CommonsenseQA | `tau/commonsense_qa` | client domain |
| CosmosQA | `allenai/cosmos_qa` | client domain |
| SocialIQA | `allenai/social_i_qa` | client domain |
| Alpaca | `tatsu-lab/alpaca` | public dataset D₀ |

The seed models (`meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.1-8B`) are also downloaded from
HuggingFace and require accepting the LLaMA license and a valid `ACCESS_TOKEN`.

### CV datasets — obtained from the CoOp repository

The CV datasets follow the setup of [CoOp (Zhou et al.)](https://github.com/kaiyangzhou/coop).
Download Pets, Flowers, EuroSAT, and ImageNet by following CoOp's
[`DATASETS.md`](https://github.com/kaiyangzhou/coop/blob/main/DATASETS.md), and use CoOp's
`split_zhou_*.json` files for the train/val splits. Arrange them under `DEFAULT_DATA_DIR` as:

```
$DEFAULT_DATA_DIR/
├── oxford-iiit-pet/                       # Pets        (client domain)
├── flowers-102/                           # Flowers102  (client domain)
├── eurosat/2750/                          # EuroSAT     (client domain)
├── imagenet_2012/                         # ImageNet    (public dataset D₀)
│   ├── train/  valid/  classnames.txt
└── zhou_data_splits/
    ├── split_zhou_OxfordPets.json
    ├── split_zhou_OxfordFlowers.json
    └── split_zhou_EuroSAT.json
```

The CLIP seed models (ViT-B/32, ViT-B/16) are downloaded automatically by the `clip` package.

## Usage

Both pipelines follow the same four steps. Example commands use the seed model `vit_b32` (CV) and
`meta-llama/Llama-3.2-3B` (NLP); `DPP` enables relevance-weighted DPP proxy selection.

### Computer vision

```bash
cd cv/scripts

# Stage 1 — select proxy data via relevance-weighted DPP, for each client domain
bash run_create_proxy_data.sh vit_b32 pets      10000 1 DPP 0
bash run_create_proxy_data.sh vit_b32 flower102 10000 1 DPP 0
bash run_create_proxy_data.sh vit_b32 eurosat   10000 1 DPP 0

# Stage 2 — proxy-aligned expert training (private data + 500 proxy samples)
bash run_train_lora.sh vit_b32 pets      10 0 0.01 500 DPP
bash run_train_lora.sh vit_b32 flower102 10 0 0.01 500 DPP
bash run_train_lora.sh vit_b32 eurosat   10 0 0.01 500 DPP

# Compute per-expert router embeddings (Eq. 7)
bash run_compute_embedding.sh vit_b32 pets      0 DPP
bash run_compute_embedding.sh vit_b32 flower102 0 DPP
bash run_compute_embedding.sh vit_b32 eurosat   0 DPP

# Stage 3 — merge experts, train the context-aware router, and evaluate on all client domains
bash run_eval_ours.sh 5 0.001 0 500 vit_b32 0.3 proxy DPP
```

### Natural language Processing

```bash
cd nlp/scripts

# Stage 1 — select proxy data via relevance-weighted DPP, for each client domain
bash run_create_proxy_data.sh meta-llama/Llama-3.2-3B tau/commonsense_qa 5000 DPP 32 1e-5
bash run_create_proxy_data.sh meta-llama/Llama-3.2-3B allenai/cosmos_qa  5000 DPP 32 1e-5
bash run_create_proxy_data.sh meta-llama/Llama-3.2-3B allenai/social_i_qa 5000 DPP 32 1e-5

# Stage 2 — proxy-aligned expert training (private data + 500 proxy samples)
bash run_lora_train_ours.sh meta-llama/Llama-3.2-3B tau/commonsense_qa  10000 None 500 DPP 16 10
bash run_lora_train_ours.sh meta-llama/Llama-3.2-3B allenai/cosmos_qa   10000 None 500 DPP 16 10
bash run_lora_train_ours.sh meta-llama/Llama-3.2-3B allenai/social_i_qa 10000 None 500 DPP 16 10

# Compute per-expert router embeddings (Eq. 7)
bash run_compute_embedding.sh meta-llama/Llama-3.2-3B tau/commonsense_qa  10000 None 500 NOBASE DPP PROXY
bash run_compute_embedding.sh meta-llama/Llama-3.2-3B allenai/cosmos_qa   10000 None 500 NOBASE DPP PROXY
bash run_compute_embedding.sh meta-llama/Llama-3.2-3B allenai/social_i_qa 10000 None 500 NOBASE DPP PROXY

# Stage 3 — merge experts, train the context-aware router, and evaluate
bash run_ours.sh meta-llama/Llama-3.2-3B 10000 -1 1e-4 1 32 500 proxy 1.0 proxy DPP
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{jiang2026metamoe,
  title     = {MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification},
  author    = {Jiang, Weisen and Chen, Shuhao and Pan, Sinno Jialin},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgements

The CV datasets and splits follow [CoOp](https://github.com/kaiyangzhou/coop). The LoRA and
PEFT components build on [HuggingFace PEFT](https://github.com/huggingface/peft).
