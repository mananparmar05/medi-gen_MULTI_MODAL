# 🏥 Multimodal Medical Report Generation

A deep learning system that automatically generates radiology reports from chest X-ray images using a multimodal architecture combining visual features, structured clinical findings, and factual consistency verification.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Dataset-IU%20X--Ray-green" alt="Dataset">
  <img src="https://img.shields.io/badge/Parameters-333M-purple" alt="Parameters">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Novel Contributions](#novel-contributions)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline](#training-pipeline)
- [Inference](#inference)
- [Results](#results)
- [Engineering Optimizations](#engineering-optimizations)
- [Configuration](#configuration)
- [Citation](#citation)

---

## Overview

This system takes **chest X-ray images** (frontal + optional lateral views) along with **14 CheXpert-aligned structured finding labels** as input, and generates **clinically accurate, fluent radiology reports** as output. The architecture introduces two novel components:

1. **Cross-Attention Bridge** — Explicitly aligns visual features with clinical findings to reduce hallucination and omission.
2. **Factual Consistency Scorer** — A BERT-based NLI (Natural Language Inference) module that penalizes the generator for producing factually inconsistent statements during training.

The system is trained using a **curriculum-weighted loss** that progressively introduces the factual consistency penalty, allowing the model to first learn fluent language generation before optimizing for medical accuracy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL REPORT GENERATOR                      │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐                               │
│  │ Frontal X-Ray│   │ Lateral X-Ray│                               │
│  └──────┬───────┘   └──────┬───────┘                               │
│         │                  │                                        │
│         ▼                  ▼                                        │
│  ┌─────────────────────────────────┐   ┌───────────────────┐       │
│  │   DenseNet-121 Vision Encoder   │   │  Metadata MLP     │       │
│  │   (Shared Weights, Dual-View)   │   │  14 → 128 → 256   │       │
│  │   Output: [B, 1024, 7, 7]      │   │  (Finding Labels)  │       │
│  └──────────────┬──────────────────┘   └────────┬──────────┘       │
│                 │                                │                  │
│                 ▼                                ▼                  │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              FiLM Fusion Layer                          │       │
│  │     γ, β = f(metadata) → modulate visual features       │       │
│  │     out = γ ⊙ features + β  (+ residual)               │       │
│  └──────────────────────┬──────────────────────────────────┘       │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         Cross-Attention Bridge (Novel #1)               │       │
│  │    14 learnable finding queries attend to visual grid    │       │
│  │    Produces finding-aware context vectors               │       │
│  │    + Weak supervision alignment loss (L_align)          │       │
│  └──────────────────────┬──────────────────────────────────┘       │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         GPT-2 Decoder (124M params)                     │       │
│  │    12 Transformer layers + injected cross-attention      │       │
│  │    Autoregressive text generation                       │       │
│  └──────────────────────┬──────────────────────────────────┘       │
│                         │                                           │
│                         ▼                                           │
│                  Generated Report                                   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │     Factual Consistency Scorer (Novel #2)               │       │
│  │    BERT-base NLI classifier (frozen during training)     │       │
│  │    Scores: entailment / contradiction / neutral          │       │
│  │    Feeds L_consist back into training loss               │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  Total Loss: L = L_gen + λ(t)·L_consist + α·L_align               │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Component Sizes

| Component | Total Params | Trainable Params |
|-----------|-------------|-----------------|
| Vision Encoder (DenseNet-121) | 9,055,104 | 2,101,248 |
| Metadata MLP | 35,456 | 35,456 |
| FiLM Fusion Layer | 594,176 | 594,176 |
| GPT-2 Decoder + Cross-Attention | 210,322,944 | 210,322,944 |
| Cross-Attention Bridge | 3,942,400 | 3,942,400 |
| NLI Scorer (BERT-base, frozen) | 109,679,875 | 0 |
| **TOTAL** | **333,629,955** | **216,996,224** |

---

## Novel Contributions

### 1. Cross-Attention Bridge
A multi-head attention module where **14 learnable finding-specific queries** attend to the visual feature grid. This explicitly aligns each clinical finding (e.g., "Cardiomegaly", "Pleural Effusion") with spatial regions of the X-ray image, reducing both **hallucination** (generating findings not present) and **omission** (missing findings that are present).

### 2. Factual Consistency Scorer (FCS)
A **BERT-based 3-way NLI classifier** that evaluates whether each sentence in a generated report is *entailed*, *contradicted*, or *neutral* with respect to the ground-truth structured findings. During training, the contradiction probability serves as a differentiable penalty signal via curriculum-weighted loss scheduling.

### 3. Curriculum-Weighted Loss
The total training objective combines three loss components:
```
L_total = L_gen + λ(t) · L_consist + α · L_align
```
- **L_gen**: Standard cross-entropy language modeling loss
- **L_consist**: NLI contradiction penalty (ramped via curriculum λ from 0 → 0.5)
- **L_align**: Weak supervision alignment loss from cross-attention bridge
- **λ(t)**: Linearly ramped from epoch 5 to epoch 15, allowing the model to learn fluent generation before introducing factual constraints

---

## Project Structure

```
MULTI_MODAL/
├── config/
│   └── config.yaml                 # All hyperparameters & paths
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # IU X-Ray dataset loader + preprocessing
│   ├── tokenizer.py                # GPT-2 tokenizer with medical special tokens
│   └── augmentation.py             # Medical image augmentations
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py           # DenseNet-121 dual-view feature extractor
│   ├── metadata_mlp.py             # Structured finding label embedder (14→256)
│   ├── film_fusion.py              # Feature-wise Linear Modulation layer
│   ├── cross_attention_bridge.py   # Novel #1: finding↔visual alignment
│   ├── decoder.py                  # GPT-2 with injected cross-attention layers
│   ├── nli_scorer.py               # Novel #2: BERT-based factual consistency
│   └── report_generator.py         # Full pipeline (assembles all components)
├── training/
│   ├── trainer.py                  # Training loop with curriculum λ warmup
│   ├── losses.py                   # Combined loss: L_gen + λ·L_consist + α·L_align
│   ├── nli_pretrainer.py           # Pre-train NLI scorer on synthetic pairs
│   └── scheduler.py                # LR scheduling + λ curriculum logic
├── evaluation/
│   ├── metrics.py                  # BLEU, ROUGE-L evaluation metrics
│   └── factual_scorer.py           # Offline factual consistency evaluation
├── utils/
│   ├── helpers.py                  # Seed, device, config, checkpoint utilities
│   └── __init__.py
├── scripts/
│   ├── download_data.py            # Download & preprocess IU X-Ray dataset
│   ├── ingest_archive.py           # Ingest from local archive
│   ├── pretrain_nli.py             # Entry point: NLI scorer pre-training
│   └── train.py                    # Entry point: main model training
├── tests/
│   └── test_pipeline.py            # End-to-end pipeline tests
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- ~4 GB disk space for model weights (DenseNet-121 + GPT-2 + BERT-base)

### Installation

```bash
# Clone the repository
git clone https://github.com/mananparmar05/medi-gen_MULTI_MODAL.git
cd medi-gen_MULTI_MODAL

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

---

## Dataset Preparation

This project uses the **IU X-Ray (Indiana University Chest X-Ray)** dataset (~3,955 reports, ~7,470 images).

### Option A: From Local Archive
```bash
python scripts/ingest_archive.py --config config/config.yaml
```

### Option B: Manual Setup
1. Download the dataset from [Open-i](https://openi.nlm.nih.gov/faq#collection)
2. Place images in `data/iu_xray/images/`
3. Place XML reports in `data/iu_xray/reports/`
4. Process annotations:
```bash
python scripts/download_data.py --process-only --config config/config.yaml
```

### Option C: Quick Development (Synthetic Data)
```bash
python scripts/download_data.py --create-sample --num-samples 50
```

---

## Training Pipeline

Training proceeds in three stages:

### Stage 1: Pre-train NLI Scorer
Pre-trains the BERT-based factual consistency scorer on synthetic finding–sentence pairs:
```bash
python scripts/pretrain_nli.py --config config/config.yaml
```
This produces `checkpoints/nli_scorer_best.pt`.

### Stage 2: Train Report Generator
Trains the full multimodal pipeline with curriculum-weighted loss:
```bash
python scripts/train.py --config config/config.yaml
```

### Resume from Checkpoint
If training is interrupted, seamlessly resume from the latest checkpoint:
```bash
python scripts/train.py --config config/config.yaml --resume checkpoints/checkpoint_latest.pt
```

### Training Phases
The training follows a multi-phase curriculum:

| Phase | Epochs | GPT-2 Body | λ (NLI Weight) | Description |
|-------|--------|-----------|----------------|-------------|
| Phase 1 | 1–3 | Frozen | 0.0 | Learn cross-attention & projection layers |
| Phase 2 | 4–5 | Unfrozen | 0.0 | Warmup with full backprop (generation only) |
| Phase 3 | 6–15 | Unfrozen | 0.0 → 0.5 | Ramp factual consistency penalty |
| Phase 4 | 16–30 | Unfrozen | 0.5 | Full training with max consistency weight |

---

## Inference

After training completes, generate reports from chest X-ray images using the best saved model:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config config/config.yaml
```

### What the Model Outputs
Given a chest X-ray image, the trained model produces:
1. **Full radiology report** — Fluent medical text describing findings
2. **Factual Consistency Score (FCS)** — Percentage of generated sentences entailed by ground-truth findings
3. **Contradiction Rate** — Percentage of factually incorrect statements

---

## Results

### Evaluation Metrics (Epoch 9 — IU X-Ray Validation Set)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **BLEU-4** | **0.1182** | ≥ 0.10 | ✅ Exceeded |
| **ROUGE-L** | **0.2150** | ≥ 0.25 | 🔄 Converging |
| **FCS** | **0.3700** | ≥ 0.75 | 🔄 Improving (λ still ramping) |
| **Val Loss** | **1.12** | — | 📉 Steadily decreasing |

### Training Convergence

| Epoch | Train Loss | Gen Loss | Val Loss | λ |
|-------|-----------|----------|----------|-----|
| 1 | 3.41 | 2.88 | — | 0.00 |
| 7 | 1.74 | 1.11 | 1.15 | 0.05 |
| 8 | 1.67 | 1.04 | 1.12 | 0.10 |
| 9 | 1.63 | 0.98 | 1.12 | 0.15 |

---

## Engineering Optimizations

### Challenge: CPU-Only Training of 333M Parameter Model
Training was conducted entirely on CPU (Apple M-series, no dedicated GPU), requiring several engineering optimizations to make the pipeline feasible:

#### 1. Keyword-Guided NLI Pre-Filtering
- **Problem**: Online NLI contradiction scoring generated ~140 BERT forward passes per training batch (14 findings × ~10 sentences), causing 68s/iter latency.
- **Solution**: Implemented keyword-based candidate pair pruning — if a finding is negative and the sentence doesn't mention any relevant clinical keywords, the pair is guaranteed neutral and BERT is skipped.
- **Impact**: 4× training speedup (68s/iter → 16s/iter) with zero loss in accuracy.

#### 2. Validation Frequency Decoupling
- **Problem**: Full autoregressive generation + NLI scoring on 549 validation samples took ~11 hours per epoch.
- **Solution**: Configured `val_metrics_every_n_epochs: 3` — fast loss-only validation every epoch (~2 min), full metric evaluation every 3rd epoch.
- **Impact**: Reduced average epoch time from ~35 hours to ~6 hours.

#### 3. Pre-Validation Checkpointing
- **Problem**: Original pipeline saved checkpoints only after validation. A validation crash (e.g., missing NLTK resource) caused total loss of trained epoch weights.
- **Solution**: Added `checkpoint_latest.pt` save immediately after `train_epoch()` completes, before validation begins.
- **Impact**: Zero compute loss on process interruption.

#### 4. Resumable Training Pipeline
- **Problem**: No mechanism to resume from saved checkpoints.
- **Solution**: Added `--resume` CLI flag that restores model weights, optimizer state, LR scheduler position, and epoch counter.
- **Impact**: Seamless recovery from any interruption.

---

## Configuration

All hyperparameters are centralized in [`config/config.yaml`](config/config.yaml):

| Category | Key Parameters |
|----------|---------------|
| **Data** | `image_size: 224`, `max_report_length: 128`, 14 CheXpert finding labels |
| **Vision** | DenseNet-121, `feature_dim: 1024`, dual-view concat mode |
| **FiLM** | `conditioning_dim: 256`, residual connection, identity init |
| **Decoder** | GPT-2 small, 12 layers, `freeze_epochs: 3` |
| **NLI** | BERT-base-uncased, 3-class classifier, frozen during generation training |
| **Training** | `batch_size: 2`, `grad_accum: 16` (effective=32), `lr: 5e-5`, cosine annealing |
| **Curriculum** | `λ_max: 0.5`, ramp epochs 5→15, early stopping patience=5 |

---

## Tech Stack

- **PyTorch** — Core deep learning framework
- **HuggingFace Transformers** — GPT-2 decoder, BERT NLI scorer
- **TorchVision** — DenseNet-121 vision encoder (ImageNet pretrained)
- **NLTK / rouge-score** — NLG evaluation metrics
- **PyYAML** — Configuration management

---

## License

This project is for academic and research purposes.

---

## Acknowledgments

- **IU X-Ray Dataset**: Indiana University, Open-i biomedical image search
- **DenseNet**: Huang et al., "Densely Connected Convolutional Networks"
- **GPT-2**: Radford et al., "Language Models are Unsupervised Multitask Learners"
- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
- **NLI for Factual Consistency**: Falke et al., "Ranking Generated Summaries by Correctness"
