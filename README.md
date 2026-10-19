# Temporal Causal-Masked Transformer for 30-Day Readmission Risk Evaluation in Longitudinal Diabetes EHR

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official code repository for the paper:  
**"Temporal Causal-Masked Transformer for 30-Day Readmission Risk Evaluation in Longitudinal Diabetes EHR"**  
Submitted to the *International Conference on Informatics and Electrical Engineering (ICIEE)*.

---

## 📌 Abstract & Overview

Predicting acute hospital readmission from multi-encounter electronic health records (EHRs) requires modeling longitudinal patient histories while preserving chronological sequence structure. This repository provides the complete experimental and evaluation codebase for the **Temporal Causal-Masked Transformer**, an autoregressive sequence model ($d_{\text{model}}=64$, $\text{nhead}=4$, 2 layers) with lower-triangular causal attention masking evaluated on the multi-hospital **UCI Diabetes 130-US Hospitals** dataset.

### Key Highlights
- **Leakage-Free Patient-Level Cohort:** 69,990 unique patients spanning 99,343 clinical encounters (after strictly filtering 2,423 in-hospital death and hospice discharge encounters).
- **Multi-Seed Benchmark:** 5 independent random seeds (42–46) evaluated on identical held-out test splits ($N = 10,499$ test patients per seed; 5.19% acute readmission prevalence).
- **Comprehensive Baseline Suite:** Benchmarked against Logistic Regression, XGBoost, Bidirectional GRU, and a non-masked Standard Transformer.
- **Statistical Rigor:** Paired two-tailed $t$-tests ($df=4$) and feature missingness stress-testing (0% to 50%).

---

## 📂 Repository Structure

```
├── checkpoints/
│   └── real_benchmark_results.json    # Pre-computed multi-seed raw evaluation tensors & metrics
├── data_uci/
│   └── diabetic_data.csv              # UCI Diabetes 130-US Hospitals dataset
├── src/
│   ├── dataset.py                     # Data loader, cohort filtering, feature scaling, & patient-level splits
│   ├── model.py                       # PyTorch models: Temporal Causal Transformer, GRU, Standard Transformer
│   ├── run_experiments.py             # 5-seed benchmark runner for all 5 models
│   ├── generate_all_verified_figures.py # Script generating Figures 1–7 directly from evaluation tensors
│   └── statistical_tests.py           # Paired t-test statistical significance calculation
├── fig1_comparative_benchmark.png     # Figure 1: Model AUROC Benchmark Bar Chart
├── fig2_loss_convergence.png          # Figure 2: Loss Convergence over 8 Epochs
├── fig3_patient_trajectory.png        # Figure 3: Authentic Multi-Encounter Progression (Patient #2448531)
├── fig4_calibration_curves.png        # Figure 4: Quantile Calibration Curves
├── fig5_roc_curves.png                # Figure 5: ROC Curves
├── fig6_precision_recall_curves.png   # Figure 6: Precision-Recall Curves
├── fig7_missing_data_robustness.png   # Figure 7: Performance Decay under Missing Data (0%–50%)
├── requirements.txt                   # Environment package dependencies
└── README.md                          # Repository documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tazkiyah-m/temporal-causal-transformer-ehr.git
   cd temporal-causal-transformer-ehr
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Provenance & Cohort Extraction

The primary dataset is the **Diabetes 130-US Hospitals for Years 1999–2008** from the UCI Machine Learning Repository ([Strack et al., 2014](https://doi.org/10.1155/2014/781670)).

### Cohort Filtering Protocol:
1. **Raw Database:** 101,766 hospital encounters across 71,518 unique patients.
2. **Mortality & Hospice Exclusions:** Encounters with `discharge_disposition_id` $\in \{11, 13, 14, 19, 20, 21\}$ (2,423 encounters across 2,399 patients) are excluded:
   - 1,528 single-encounter patients with terminal outcomes are completely removed.
   - 871 patients with historical admissions preceding a terminal event retain their valid prior visits; only the terminal encounter is truncated.
3. **Cleaned Cohort:** **69,990 unique patients** and **99,343 clinical encounters**.
4. **Target:** Acute 30-day readmission (`<30`) following the final longitudinal visit (4.96% cohort prevalence, 5.19% held-out test prevalence).

---

## 🚀 Running Experiments & Reproducing Results

### 1. Run the Multi-Seed Benchmark (5 Seeds: 42, 43, 44, 45, 46)
To train all 5 models across all 5 seeds from scratch and compute test set metrics:
```bash
python src/run_experiments.py
```
This updates `checkpoints/real_benchmark_results.json`.

### 2. Run Statistical Tests
To execute paired two-tailed $t$-tests ($df=4$) across seeds:
```bash
python src/statistical_tests.py
```

### 3. Generate All Manuscript Figures (Figures 1–7)
To regenerate all 7 publication-ready figures directly from model evaluation outputs:
```bash
python src/generate_all_verified_figures.py
```

---

## 📈 Benchmark Results Summary

Evaluated on held-out test sets ($N = 10,499$ patients per seed, 5 independent seeds, Mean ± SD):

| Architecture | Category | AUROC (Mean ± SD) | AUPRC (Mean ± SD) | F1-Score (Mean ± SD) | Brier Score (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Tabular Linear | $0.6955 \pm 0.0049$ | $0.1295 \pm 0.0082$ | $0.2022 \pm 0.0131$ | $0.2153 \pm 0.0017$ |
| **XGBoost** | Tabular GBDT | $0.7051 \pm 0.0093$ | $\mathbf{0.1456 \pm 0.0160}$ | $\mathbf{0.2215 \pm 0.0154}$ | $0.2012 \pm 0.0012$ |
| **Bidirectional GRU** | Recurrent Sequence | $0.6834 \pm 0.0059$ | $0.1248 \pm 0.0099$ | $0.1849 \pm 0.0152$ | $0.0686 \pm 0.0165$ |
| **Standard Transformer** | Ablation (No Mask) | $0.7012 \pm 0.0064$ | $0.1395 \pm 0.0092$ | $0.2139 \pm 0.0112$ | $\mathbf{0.0611 \pm 0.0124}$ |
| **Temporal Causal Transformer** | Proposed Sequence | $\mathbf{0.7078 \pm 0.0081}$ | $0.1389 \pm 0.0122$ | $0.2175 \pm 0.0119$ | $0.0710 \pm 0.0346$ |

*Key Takeaway: The proposed transformer achieves discriminative parity with tuned tree ensembles (XGBoost, $p = 0.6171$) and significantly outperforms recurrent baselines (GRU, $p = 0.0005$).*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
