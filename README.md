# 🧠 Parkinson's Disease Detection from Sustained Vowel Phonations

### TF-NMF Acoustic Features • TLBO Feature Selection • RBF-SVM Classification

An end-to-end biomedical speech-processing and machine-learning pipeline for detecting **Parkinson's Disease (PD)** from sustained vowel phonations.

The framework extracts **29 acoustic features** from each `.wav` recording, selects the **5 most discriminative features** using **Teaching-Learning-Based Optimization (TLBO)**, and classifies subjects using an **RBF-kernel Support Vector Machine (SVM)** with **Stratified 5-Fold Cross-Validation**.

---

## 📌 Overview

Parkinson's Disease can affect speech production through **hypokinetic dysarthria**, producing measurable changes in vocal stability, spectral characteristics, energy distribution, and temporal dynamics.

This project investigates whether these acoustic changes can be detected from sustained vowel phonations:

**/A/, /E/, /I/, /O/, /U/**

### Complete Pipeline

    Sustained Vowel Audio (.wav)
                  │
                  ▼
        ┌──────────────────────┐
        │ Feature Extraction  │
        └──────────┬───────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    13 MFCC             STFT Magnitude
    Features                 │
                             ▼
                      NMF Decomposition
                           K = 4
                             │
                             ▼
                     4 × 4 Descriptors
                       = 16 Features
        │                     │
        └──────────┬──────────┘
                   ▼
             29-D Feature Vector
                   │
                   ▼
        ┌──────────────────────┐
        │ TLBO Feature         │
        │ Selection 29 → 5     │
        └──────────┬───────────┘
                   │
                   ▼
          Z-Score Normalization
                   │
                   ▼
        ┌──────────────────────┐
        │ RBF-SVM Classifier   │
        │ Stratified 5-Fold CV │
        └──────────┬───────────┘
                   │
                   ▼
          Diagnostic Metrics
                   │
                   ▼
        Multi-Vowel Soft Voting
                   │
                   ▼
          Subject-Level Result

---

## ✨ Key Features

- 🎙️ Sustained-vowel acoustic analysis
- 🔊 13 Mel-Frequency Cepstral Coefficients (MFCC)
- 📊 16 Time-Frequency NMF (TF-NMF) descriptors
- 🧮 29 → 5 dimensional feature selection
- 🧑‍🏫 Teaching-Learning-Based Optimization (TLBO)
- 📐 Scatter-ratio-based feature-selection objective
- ⚖️ Z-score feature normalization
- 🤖 RBF-kernel Support Vector Machine
- 🔀 Stratified 5-Fold Cross-Validation
- 🩺 Sensitivity and specificity analysis
- 📈 Accuracy, Precision, F1-score and MCC
- 🗣️ Multi-vowel subject-level soft voting

---

# 📁 Repository Structure

    parkinsons-tlbo-detection/
    │
    ├── data/
    │   ├── Vowel_Healthy/
    │   │   ├── A/
    │   │   ├── E/
    │   │   ├── I/
    │   │   ├── O/
    │   │   └── U/
    │   │
    │   └── Vowel_PD/
    │       ├── A/
    │       ├── E/
    │       ├── I/
    │       ├── O/
    │       └── U/
    │
    ├── src/
    │   ├── __init__.py
    │   ├── feature_extraction.py
    │   ├── tlbo_optimizer.py
    │   └── evaluate.py
    │
    ├── outputs/
    │   ├── figures/
    │   ├── reports/
    │   └── results/
    │
    ├── THEORY.md
    ├── main.py
    ├── requirements.txt
    └── README.md

---

# 📊 Dataset Organization

The dataset contains sustained vowel recordings divided into two diagnostic classes:

    data/
    │
    ├── Vowel_Healthy/
    │   ├── A/     # 29 recordings
    │   ├── E/     # 29 recordings
    │   ├── I/     # 29 recordings
    │   ├── O/     # 29 recordings
    │   └── U/     # 29 recordings
    │
    └── Vowel_PD/
        ├── A/     # 30 recordings
        ├── E/     # 30 recordings
        ├── I/     # 30 recordings
        ├── O/     # 30 recordings
        └── U/     # 30 recordings

For each vowel:

$$
N_H = 29
$$

$$
N_{PD} = 30
$$

Therefore:

$$
N = N_H + N_{PD} = 59
$$

Each recording is transformed into a 29-dimensional acoustic vector:

$$
\boxed{\mathbf{x}\in\mathbb{R}^{29}}
$$

> **Important:** The mapping between recordings and subjects must be preserved during evaluation. Multiple recordings or vowels belonging to the same subject should not be split between training and test partitions.

---

# 🔬 Methodology

## 1. Acoustic Feature Extraction

Each audio recording is represented using two complementary feature families:

$$
\boxed{
29 = 13\text{ MFCC} + 16\text{ TF-NMF}
}
$$

| Feature Family | Number of Features | Main Information |
|---|---:|---|
| MFCC | 13 | Spectral and cepstral characteristics |
| TF-NMF | 16 | Time-frequency and temporal characteristics |
| **Total** | **29** | **Combined acoustic representation** |

---

# 🎵 2. MFCC Feature Extraction

Mel-Frequency Cepstral Coefficients describe the spectral characteristics of speech using the perceptually motivated Mel frequency scale.

The Mel transformation is:

$$
\boxed{
m(f)=2595\log_{10}\left(1+\frac{f}{700}\right)
}
$$

### MFCC Processing

    Audio Signal
         ↓
    Framing & Windowing
         ↓
    FFT
         ↓
    Power Spectrum
         ↓
    40 Mel Filter Banks
         ↓
    Log Filter-Bank Energies
         ↓
    Discrete Cosine Transform
         ↓
    13 MFCC Features

For the $n$-th cepstral coefficient:

$$
\boxed{
c_n =
\sum_{m=1}^{M}
\log(Y_m)
\cos
\left[
\frac{\pi n(m-\frac{1}{2})}{M}
\right]
}
$$

where:

- $Y_m$ is the energy of the $m$-th Mel filter.
- $M$ is the number of Mel filters.
- $c_n$ is the $n$-th MFCC coefficient.

The first 13 coefficients are retained:

$$
\mathbf{x}_{MFCC}
=
[c_0,c_1,\ldots,c_{12}]
$$

Frame-level coefficients are aggregated to produce one 13-dimensional feature vector per recording.

---

# 🕒 3. Time-Frequency NMF Features

MFCCs primarily describe spectral characteristics. To capture additional time-frequency information, the framework applies **Non-Negative Matrix Factorization (NMF)** to the STFT magnitude spectrogram.

---

## 3.1 Short-Time Fourier Transform

The audio signal is transformed into a time-frequency representation:

$$
x[n]\longrightarrow X(f,t)
$$

The magnitude spectrogram is:

$$
\boxed{
V=|X(f,t)|
}
$$

where:

$$
V\in\mathbb{R}_{\geq0}^{F\times T}
$$

and:

- $F$ = number of frequency bins
- $T$ = number of time frames

---

## 3.2 NMF Decomposition

The non-negative spectrogram is factorized as:

$$
\boxed{
V\approx WH
}
$$

where:

$$
W\in\mathbb{R}_{\geq0}^{F\times K}
$$

and:

$$
H\in\mathbb{R}_{\geq0}^{K\times T}
$$

The number of NMF components is:

$$
\boxed{K=4}
$$

Therefore:

$$
V_{F\times T}
\approx
W_{F\times4}
H_{4\times T}
$$

The NMF decomposition minimizes the reconstruction error:

$$
\boxed{
\mathcal{L}(W,H)
=
\left\|V-WH\right\|_F^2
}
$$

subject to:

$$
W\geq0,\qquad H\geq0
$$

The matrix $H$ contains the temporal activations of the four NMF components.

---

# 📐 4. NMF Temporal Descriptors

For each NMF component:

$$
k\in\{1,2,3,4\}
$$

four temporal descriptors are calculated.

Therefore:

$$
4\text{ components}
\times
4\text{ descriptors}
=
\boxed{16\text{ TF-NMF features}}
$$

---

## 4.1 Mean Activation

For component $k$:

$$
\boxed{
\mu_k=
\frac{1}{T}
\sum_{t=1}^{T}
H_{k,t}
}
$$

This represents the average temporal activation of the component.

---

## 4.2 Standard Deviation

$$
\boxed{
\sigma_k=
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
(H_{k,t}-\mu_k)^2
}
}
$$

This measures the temporal variation of the component.

---

## 4.3 Discontinuity

Temporal discontinuity is calculated as:

$$
\boxed{
\Delta_k=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
\left|
H_{k,t+1}-H_{k,t}
\right|
}
$$

Higher values indicate stronger frame-to-frame changes.

---

## 4.4 Hoyer Sparsity

Hoyer sparsity measures the concentration of the activation vector:

$$
\boxed{
S_k=
\frac{
\sqrt{T}
-
\frac{\|H_k\|_1}{\|H_k\|_2}
}{
\sqrt{T}-1
}
}
$$

where:

$$
\|H_k\|_1
=
\sum_t |H_{k,t}|
$$

and:

$$
\|H_k\|_2
=
\sqrt{\sum_t H_{k,t}^2}
$$

The value approximately satisfies:

$$
0\leq S_k\leq1
$$

Higher values indicate greater sparsity.

---

# 🧩 5. Complete 29-Dimensional Representation

The complete feature vector is obtained by concatenating MFCC and TF-NMF features:

$$
\boxed{
\mathbf{x}
=
[
\mathbf{x}_{MFCC},
\mathbf{x}_{NMF}
]
}
$$

where:

$$
\mathbf{x}_{MFCC}\in\mathbb{R}^{13}
$$

and:

$$
\mathbf{x}_{NMF}\in\mathbb{R}^{16}
$$

Therefore:

$$
\boxed{
\mathbf{x}\in\mathbb{R}^{29}
}
$$

For 59 recordings:

$$
\boxed{
X\in\mathbb{R}^{59\times29}
}
$$

---

# 🧑‍🏫 6. TLBO Feature Selection

The original 29-dimensional feature space is reduced to five features using **Teaching-Learning-Based Optimization (TLBO)**.

The optimization problem is:

$$
\boxed{
\mathbf{s}^{*}
=
\arg\min_{\mathbf{s}}J(\mathbf{s})
}
$$

where:

$$
\mathbf{s}
\subset
\{1,2,\ldots,29\}
$$

subject to:

$$
|\mathbf{s}|=5
$$

Thus:

$$
\boxed{
29\rightarrow5
}
$$

The objective function evaluates the discriminative quality of each candidate feature subset.

---

# 📉 7. Scatter Ratio Objective

For the two diagnostic classes:

$$
C\in\{H,PD\}
$$

the class centroid is:

$$
\boxed{
\boldsymbol{\mu}_c
=
\frac{1}{N_c}
\sum_{\mathbf{x}\in c}\mathbf{x}
}
$$

The overall centroid is:

$$
\boxed{
\boldsymbol{\mu}
=
\frac{1}{N}
\sum_{i=1}^{N}\mathbf{x}_i
}
$$

---

## 7.1 Within-Class Scatter

The within-class scatter is:

$$
\boxed{
S_w=
\sum_c
\sum_{\mathbf{x}\in c}
\left\|
\mathbf{x}-\boldsymbol{\mu}_c
\right\|_2^2
}
$$

A smaller $S_w$ indicates that samples from the same diagnostic class are more compact.

---

## 7.2 Between-Class Scatter

The between-class scatter is:

$$
\boxed{
S_b=
\sum_c
N_c
\left\|
\boldsymbol{\mu}_c-\boldsymbol{\mu}
\right\|_2^2
}
$$

A larger $S_b$ indicates greater separation between the diagnostic classes.

---

## 7.3 Optimization Objective

The TLBO fitness function is:

$$
\boxed{
J=
\frac{S_w}{S_b+\epsilon}
}
$$

where $\epsilon$ is a small positive value used for numerical stability.

The optimization therefore attempts to:

$$
\downarrow S_w
$$

while simultaneously achieving:

$$
\uparrow S_b
$$

Hence:

$$
\boxed{
\min\frac{S_w}{S_b+\epsilon}
}
$$

selects features that produce compact classes with strong inter-class separation.

---

# 🧑‍🏫 8. TLBO Optimization Phases

TLBO consists of two main phases:

    Initial Population
           │
           ▼
      Fitness Evaluation
           │
           ▼
      Select Best Learner
           │
           ▼
    ┌──────┴──────┐
    │             │
    ▼             ▼
Teacher Phase  Learner Phase
    │             │
    └──────┬──────┘
           │
           ▼
    Updated Population
           │
           ▼
      Best Feature Set

---

## 8.1 Teacher Phase

Let:

$$
X_i
$$

represent the $i$-th learner.

The best learner is selected as the teacher:

$$
X_T
$$

The class mean is:

$$
\boxed{
M=
\frac{1}{P}
\sum_{i=1}^{P}X_i
}
$$

The teacher-phase update is:

$$
\boxed{
X_i'
=
X_i+
r
\left(
X_T-T_FM
\right)
}
$$

where:

$$
r\sim U(0,1)
$$

and:

$$
T_F\in\{1,2\}
$$

is the teaching factor.

---

## 8.2 Learner Phase

Two learners $X_i$ and $X_j$ interact.

If learner $j$ has better fitness:

$$
J(X_j)<J(X_i)
$$

then learner $i$ moves toward learner $j$:

$$
\boxed{
X_i'
=
X_i+
r(X_j-X_i)
}
$$

Otherwise:

$$
\boxed{
X_i'
=
X_i+
r(X_i-X_j)
}
$$

After each update, candidate solutions are constrained to the valid feature-selection space.

The best feature subset obtained during optimization is retained.

---

# 🤖 9. RBF-SVM Classification

After TLBO feature selection, each recording is represented using five selected features:

$$
\boxed{
X\in\mathbb{R}^{59\times5}
}
$$

An SVM with a **Radial Basis Function (RBF)** kernel is used for classification.

The RBF kernel is:

$$
\boxed{
K(\mathbf{x}_i,\mathbf{x}_j)
=
\exp
\left(
-\gamma
\|\mathbf{x}_i-\mathbf{x}_j\|_2^2
\right)
}
$$

where:

- $C$ controls regularization.
- $\gamma$ controls the RBF kernel width.

The binary labels are:

$$
y_i\in\{-1,+1\}
$$

The SVM dual optimization problem is:

$$
\boxed{
\max_{\boldsymbol{\alpha}}
\left[
\sum_{i=1}^{N}\alpha_i
-
\frac{1}{2}
\sum_{i=1}^{N}
\sum_{j=1}^{N}
\alpha_i\alpha_jy_iy_j
K(\mathbf{x}_i,\mathbf{x}_j)
\right]
}
$$

subject to:

$$
0\leq\alpha_i\leq C
$$

and:

$$
\boxed{
\sum_{i=1}^{N}\alpha_i y_i=0
}
$$

---

# ⚖️ 10. Z-Score Feature Scaling

Feature normalization is performed within each cross-validation fold.

For feature $j$:

$$
\boxed{
z_{ij}
=
\frac{x_{ij}-\mu_j^{train}}
{\sigma_j^{train}}
}
$$

where:

$$
\mu_j^{train}
=
\frac{1}{N_{train}}
\sum_i x_{ij}
$$

and:

$$
\sigma_j^{train}
=
\sqrt{
\frac{1}{N_{train}}
\sum_i
(x_{ij}-\mu_j^{train})^2
}
$$

The test fold is transformed using the parameters calculated from the training fold.

This prevents test-set information from leaking into the training process.

---

# 🔄 11. Stratified 5-Fold Cross-Validation

The dataset is divided into five stratified folds:

$$
\boxed{k=5}
$$

Each fold maintains approximately the original Healthy/PD class distribution.

For each iteration:

    Training Set:
    4 folds

    Test Set:
    1 fold

The procedure is repeated five times so that every fold is used exactly once as the test set.

---

# 🩺 12. Diagnostic Performance Metrics

The classifier is evaluated using the following confusion matrix:

| | Predicted Healthy | Predicted PD |
|---|---:|---:|
| **Actual Healthy** | TN | FP |
| **Actual PD** | FN | TP |

---

## 12.1 Accuracy

$$
\boxed{
Accuracy=
\frac{TP+TN}
{TP+TN+FP+FN}
}
$$

Accuracy represents the proportion of correctly classified samples.

---

## 12.2 Sensitivity

Sensitivity measures the proportion of PD samples correctly identified:

$$
\boxed{
Sensitivity=
\frac{TP}{TP+FN}
}
$$

Sensitivity is also known as **Recall** or **True Positive Rate**.

---

## 12.3 Specificity

Specificity measures the proportion of Healthy samples correctly identified:

$$
\boxed{
Specificity=
\frac{TN}{TN+FP}
}
$$

---

## 12.4 Precision

$$
\boxed{
Precision=
\frac{TP}{TP+FP}
}
$$

Precision measures how many samples predicted as PD are actually PD.

---

## 12.5 F1-Score

$$
\boxed{
F1=
2
\frac{
Precision\times Sensitivity
}{
Precision+Sensitivity
}
}
$$

The F1-score provides a balance between precision and sensitivity.

---

## 12.6 Matthews Correlation Coefficient

The Matthews Correlation Coefficient is:

$$
\boxed{
MCC=
\frac{
TP\cdot TN-FP\cdot FN
}{
\sqrt{
(TP+FP)
(TP+FN)
(TN+FP)
(TN+FN)
}
}
}
$$

The MCC range is:

$$
-1\leq MCC\leq1
$$

Interpretation:

| MCC | Interpretation |
|---:|---|
| $1$ | Perfect prediction |
| $0$ | Random-level association |
| $-1$ | Completely incorrect prediction |

---

# 🗣️ 13. Multi-Vowel Subject-Level Ensemble

The system can evaluate each vowel independently and then combine the predictions at the subject level.

For a subject with available vowels:

$$
\{A,E,I,O,U\}
$$

each vowel model produces a probability of Parkinson's Disease:

$$
P_A(PD),
P_E(PD),
P_I(PD),
P_O(PD),
P_U(PD)
$$

The soft-voting ensemble probability is:

$$
\boxed{
P_{ensemble}(PD)
=
\frac{1}{V}
\sum_{v=1}^{V}
P_v(PD)
}
$$

where $V$ is the number of available vowels.

The final subject-level prediction is:

$$
\boxed{
\hat{y}=
\begin{cases}
PD,
&
P_{ensemble}(PD)\geq\tau
\\[6pt]
Healthy,
&
P_{ensemble}(PD)<\tau
\end{cases}
}
$$

with:

$$
\boxed{\tau=0.5}
$$

as the default classification threshold.

This approach combines evidence from multiple sustained vowels rather than relying on a single vowel.

---

# 🧪 14. Complete Mathematical Formulation

The complete pipeline can be expressed as:

### Step 1 — Audio Representation

$$
x(t)
\overset{\text{MFCC + STFT/NMF}}{\longrightarrow}
\mathbf{z}\in\mathbb{R}^{29}
$$

### Step 2 — Feature Selection

$$
\mathbf{z}
\overset{\text{TLBO}}{\longrightarrow}
\mathbf{z}_S\in\mathbb{R}^{5}
$$

### Step 3 — Feature Normalization

$$
\mathbf{z}_S
\overset{\text{Z-score}}{\longrightarrow}
\tilde{\mathbf{z}}_S
$$

### Step 4 — Classification

$$
\tilde{\mathbf{z}}_S
\overset{\text{RBF-SVM}}{\longrightarrow}
P(y\mid\tilde{\mathbf{z}}_S)
$$

### Step 5 — Subject-Level Fusion

$$
\{P_A,P_E,P_I,P_O,P_U\}
\overset{\text{Soft Voting}}{\longrightarrow}
P_{subject}(PD)
$$

### Overall Pipeline

$$
\boxed{
\text{Audio}
\rightarrow
\text{MFCC + TF-NMF}
\rightarrow
29D
\rightarrow
\text{TLBO}
\rightarrow
5D
\rightarrow
\text{RBF-SVM}
\rightarrow
\text{Vowel Prediction}
\rightarrow
\text{Soft Voting}
\rightarrow
\text{Subject Prediction}
}
$$

---

# 🚀 Installation

## 1. Clone the Repository

    git clone https://github.com/<YOUR_GITHUB_USERNAME>/parkinsons-tlbo-detection.git
    cd parkinsons-tlbo-detection

## 2. Create a Virtual Environment

### Windows CMD

    python -m venv venv
    venv\Scripts\activate

### Windows PowerShell

    python -m venv venv
    .\venv\Scripts\Activate.ps1

### Linux / macOS

    python3 -m venv venv
    source venv/bin/activate

## 3. Install Dependencies

    pip install -r requirements.txt

---

# ▶️ Running the Pipeline

Place the `.wav` files into the appropriate dataset directories.

Then execute:

    python main.py

The program performs the following operations:

    1. Load audio recordings
    2. Extract MFCC features
    3. Compute STFT
    4. Perform NMF decomposition
    5. Extract NMF temporal descriptors
    6. Construct the 29-dimensional feature matrix
    7. Perform TLBO feature selection
    8. Select five optimal features
    9. Apply feature normalization
    10. Train the RBF-SVM
    11. Perform Stratified 5-Fold Cross-Validation
    12. Calculate diagnostic metrics
    13. Aggregate vowel-level predictions
    14. Generate subject-level ensemble results

---

# 📄 Expected Outputs

Results are stored under:

    outputs/
    ├── figures/
    ├── reports/
    └── results/

Typical output files may include:

    outputs/
    ├── figures/
    │   ├── confusion_matrix.png
    │   └── feature_analysis.png
    │
    ├── reports/
    │   ├── A_results.csv
    │   ├── E_results.csv
    │   ├── I_results.csv
    │   ├── O_results.csv
    │   └── U_results.csv
    │
    └── results/
        ├── vowel_summary.csv
        └── subject_ensemble.csv

---

# 🔬 Reproducibility and Experimental Validity

For scientifically reliable evaluation, several considerations are important.

## Feature Selection

TLBO should ideally be performed using **training data only inside each cross-validation split** when estimating generalization performance.

If TLBO is run on the complete dataset before cross-validation, information from the test folds influences feature selection.

This can produce optimistic performance estimates.

## Feature Scaling

The mean and standard deviation used for Z-score normalization should be calculated from the training partition only.

The test partition should then be transformed using those training parameters.

## Subject Independence

If several vowel recordings belong to the same subject, all recordings from that subject should remain in the same evaluation partition.

Otherwise, the same subject can contribute recordings to both training and testing, resulting in data leakage.

## Randomness

TLBO is a stochastic optimization algorithm.

For reproducible experiments, random seeds should be controlled whenever possible.

---

# ⚠️ Important Methodological Consideration

There are two possible experimental designs.

## Design A — Global TLBO Selection

    Complete Dataset
          │
          ▼
        TLBO
          │
          ▼
    5 Selected Features
          │
          ▼
     5-Fold CV
          │
          ▼
        SVM

This design is useful for exploratory analysis.

However, the reported cross-validation performance may be **optimistically biased**, because feature selection has already seen the complete dataset.

---

## Design B — Fold-Wise / Nested TLBO Selection

    Outer Cross-Validation
             │
             ├── Training Data
             │       │
             │       ▼
             │      TLBO
             │       │
             │       ▼
             │   5 Features
             │       │
             │       ▼
             │      SVM
             │       │
             │       ▼
             │   Test Fold
             │
             └── Repeat for every fold

This is the preferred design for obtaining an unbiased estimate of generalization performance.

For publication-quality experiments, **TLBO feature selection should be performed inside the training portion of each outer cross-validation fold**.

---

# 📚 Theory

The mathematical foundations of the project are documented separately in:

    THEORY.md

The theory document covers:

- Parkinsonian speech characteristics
- MFCC computation
- Mel-frequency transformation
- STFT
- Non-Negative Matrix Factorization
- NMF temporal descriptors
- Hoyer sparsity
- Within-class scatter
- Between-class scatter
- TLBO optimization
- RBF-SVM
- Cross-validation
- Diagnostic performance metrics

---

# 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| SciPy | Signal processing |
| Librosa | Audio processing |
| scikit-learn | SVM and cross-validation |
| Pandas | Data management |
| Matplotlib | Visualization |

Exact dependency versions are specified in:

    requirements.txt

---

# 📈 Research Objective

The central hypothesis of the framework is:

$$
\boxed{
\text{Acoustic characteristics of sustained vowels}
\Rightarrow
\text{Discriminative information for PD detection}
}
$$

The proposed system investigates whether combining:

$$
\boxed{
\text{MFCC}+\text{TF-NMF}
}
$$

provides a richer acoustic representation than using either feature family independently.

The original feature space is:

$$
29\text{ features}
$$

which is reduced to:

$$
\boxed{5\text{ selected features}}
$$

using TLBO.

The resulting compact representation is then classified using an RBF-SVM.

---

# 📊 Pipeline Summary

| Stage | Input | Operation | Output |
|---|---|---|---|
| Audio | `.wav` | Audio loading | Audio signal |
| MFCC | Audio | Mel filtering + DCT | 13 features |
| STFT | Audio | Time-frequency transform | Spectrogram |
| NMF | Spectrogram | $V\approx WH$ | 4 components |
| TF-NMF | $H$ | 4 descriptors/component | 16 features |
| Fusion | MFCC + NMF | Concatenation | 29 features |
| TLBO | 29 features | Scatter-ratio optimization | 5 features |
| Scaling | 5 features | Z-score normalization | Normalized features |
| SVM | 5 features | RBF classification | Class probabilities |
| CV | Dataset | Stratified 5-Fold | Performance metrics |
| Ensemble | Vowel predictions | Soft voting | Subject prediction |

---

# 🎯 Final Summary

The proposed Parkinson's Disease detection framework integrates:

- Speech signal processing
- MFCC acoustic analysis
- STFT time-frequency analysis
- Non-Negative Matrix Factorization
- Temporal NMF descriptors
- Teaching-Learning-Based Optimization
- Discriminative feature selection
- RBF-kernel SVM classification
- Stratified cross-validation
- Diagnostic performance metrics
- Multi-vowel subject-level fusion

The complete framework is:

$$
\boxed{
\text{Sustained Vowel Audio}
\rightarrow
\text{MFCC + TF-NMF}
\rightarrow
\text{29 Features}
\rightarrow
\text{TLBO}
\rightarrow
\text{5 Features}
\rightarrow
\text{RBF-SVM}
\rightarrow
\text{Vowel Prediction}
\rightarrow
\text{Soft Voting}
\rightarrow
\text{Subject Prediction}
}
$$

---

# ⚕️ Disclaimer

This project is intended for **research and educational purposes only**.

It is **not a clinically validated diagnostic system** and should not be used as a substitute for professional medical examination, diagnosis, or treatment.

---

# 📜 License

Add your preferred open-source license here, for example:

    MIT License

See the `LICENSE` file for complete license terms.

---

# ⭐ Citation

If this project is used in academic work, please cite the corresponding paper or repository.

Example:

    @software{parkinsons_tlbo_detection,
      title  = {Parkinson's Disease Detection via TF-NMF and TLBO Feature Selection},
      author = {YOUR NAME},
      year   = {2026},
      url    = {https://github.com/<YOUR_GITHUB_USERNAME>/parkinsons-tlbo-detection}
    }

---

# 🙏 Acknowledgements

This project builds upon established methods in:

- Biomedical speech processing
- Mel-Frequency Cepstral Coefficients
- Short-Time Fourier Transform
- Non-Negative Matrix Factorization
- Teaching-Learning-Based Optimization
- Support Vector Machines
- Statistical pattern recognition

---

## ⚕️ Research Note

The system is designed as an experimental machine-learning framework for investigating acoustic biomarkers associated with Parkinson's Disease.

Performance should be interpreted in the context of:

- Dataset size
- Subject independence
- Cross-validation methodology
- Feature-selection strategy
- Class balance
- Recording conditions
- Hyperparameter selection
- External validation

For clinical translation, independent datasets and prospective clinical validation would be required.
