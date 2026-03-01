<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Sequence-Based%20Prediction%20of%20Chromatin%20Looping&fontSize=32&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=DNABERT-2%20%E2%9C%A6%20Transformer%20Genomics%20%E2%9C%A6%203D%20Genome%20Modeling&descAlignY=60&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![BioPython](https://img.shields.io/badge/BioPython-Genomics-4CAF50?style=for-the-badge)](https://biopython.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Modeling-000000?style=for-the-badge)](https://networkx.org)

<br/>

> **Transformer-Based Genomic Modeling**
> *Predicting 3D chromatin interactions directly from raw DNA sequence*

<br/>

---

</div>

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🧬 Biological Background](#-biological-background)
- [💡 Research Gap](#-research-gap)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [🧠 Methodology](#-methodology)
- [📊 Dataset Construction](#-dataset-construction)
- [🤖 Model Architecture (DNABERT-2)](#-model-architecture-dnabert-2)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [📊 Results](#-results)
- [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
- [🚀 Running the Project](#-running-the-project)
- [🔭 Future Scope](#-future-scope)

---

## 🎯 Overview

Chromatin looping enables long-range DNA interactions between enhancers and promoters, playing a critical role in gene regulation and disease mechanisms.

Traditional lab techniques such as **Hi-C** and **ChIA-PET** are expensive and time-consuming.

This project proposes a **purely sequence-based deep learning approach** using **DNABERT-2**, a transformer model pretrained on genomic sequences, to predict chromatin looping interactions directly from raw DNA sequence.

---

## 🧬 Biological Background

Chromatin looping allows:

```
Enhancer ───────────────► Promoter
     (Distant DNA regions interact in 3D space)
```

Disruptions in these interactions are linked to:

- Cancer
- Genetic disorders
- Regulatory dysfunction

Predicting loops computationally reduces reliance on costly wet-lab experiments.

---

## 💡 Research Gap

Existing methods:

- Depend on spatial genomic data (Hi-C, ChIA-PET)
- Use hybrid ML pipelines (e.g., DNABERT + SVM/RF)
- Struggle with long-range modeling

This project:

- Uses **DNABERT-2**
- Implements **end-to-end fine-tuning**
- Leverages **ALiBi positional encoding**
- Avoids multi-model fusion complexity

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────┐
│        ChIA-PET Interaction Data        │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│  Looping & Non-Looping Dataset Creation │
│  (Graph-based filtering using NetworkX) │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│      DNA Sequence Extraction (FASTA)    │
│              using BioPython            │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│        DNABERT-2 Fine-Tuning            │
│   Chromosome-wise Binary Classification │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│        Evaluation & Heatmaps            │
└──────────────────────────────────────────┘
```

---

## 🧠 Methodology

### 1️⃣ Looping Regions

- Extracted from ChIA-PET datasets
- Filtered by interaction score ≥ 4
- Divided into 60,000 bp windows

### 2️⃣ Non-Looping Regions

Two graph-based approaches:

**Approach 1:** Complement Graph  
- Build looping graph  
- Generate complement graph  
- Extract non-interacting region pairs  

**Approach 2:** Adjacent Gap Extraction  
- Identify spatial gaps between looping regions  
- Apply biological filtering  

Short regions (<256 bp) removed.

---

## 📊 Dataset Construction

- Genome source: Human FASTA files
- Loop annotations: ChIA-PET (ENCODE / 4D Nucleome)
- 22 chromosomes modeled independently
- Balanced binary labels:
  - 1 → Looping
  - 0 → Non-looping

---

## 🤖 Model Architecture (DNABERT-2)

### Key Components

- Byte Pair Encoding (BPE) tokenizer
- ALiBi positional encoding
- Multi-head self-attention
- Transformer encoder stack
- Classification head on `[CLS]` token

### Fine-Tuning Strategy

- Chromosome-wise training
- HuggingFace Trainer API
- Partial layer fine-tuning
- Mixed precision (fp16)

---

## 📈 Evaluation Metrics

- **Accuracy**
- **F1 Score**
- **Matthews Correlation Coefficient (MCC)**
- **AUPRC (Area Under Precision-Recall Curve)**

AUPRC was particularly important due to dataset imbalance.

---

## 📊 Results

Key Observations:

- Strong diagonal dominance in chromosome heatmaps
- High AUPRC across most chromosomes
- chr2, chr5, chr10, chr18 showed strong MCC
- chr21 and chr22 exhibited relatively lower scores

The model demonstrates:

- Effective long-range sequence modeling
- Cross-chromosomal generalization
- Biological interpretability via attention

---

## 🛠️ Tech Stack

```python
# Deep Learning
PyTorch
HuggingFace Transformers

# Genomics
BioPython

# Graph Modeling
NetworkX

# Data Processing
NumPy
Pandas
Scikit-learn

# Metrics
Torchmetrics
```

---

## 🚀 Running the Project

### Fine-Tune DNABERT-2

```
python train.py --chromosome chr8
```

### Evaluate Model

```
python evaluate.py --model saved_model_path
```

### Generate Heatmaps

```
python visualize_results.py
```

---

## 🔭 Future Scope

```
⚡ Flash Attention integration
🧠 Larger input sequence lengths
🖥️ Training on full-genome scale with high-memory GPUs
🌐 Multi-chromosome unified modeling
🔬 Integration with epigenomic signals
```

---

<div align="center">

**Transformer Models for 3D Genomics**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
