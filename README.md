# PSA-SPC-DM: Pulse Shape Analysis for Dark Matter Detection

A machine learning pipeline for discriminating dark matter signals from background noise in single-phase charge (SPC) detectors using pulse shape analysis (PSA) and physics-based feature extraction.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Physics Background](#physics-background)
- [Solution Overview](#solution-overview)
- [Methodology](#methodology)
- [Results](#results)
- [Relevance to Physics](#relevance-to-physics)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Educational Notebooks](#educational-notebooks)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

Dark matter detection experiments like NEWS-G and DarkSPHERE use single-phase charge (SPC) detectors to search for Weakly Interacting Massive Particles (WIMPs). The challenge is **distinguishing rare dark matter signal events from background noise and detector artifacts**, particularly:

1. **Background events**: Electronic recoils with different pulse shapes than nuclear recoils
2. **Pile-up events**: Multiple particles arriving at the detector simultaneously, creating distorted waveforms
3. **Instrumental noise**: Systematic effects that mimic signal characteristics

A false positive discovery would be catastrophic, so robust background discrimination is critical.

---

## Physics Background

### SPC Detectors

Single-Phase Charge (SPC) detectors collect ionization produced by particle interactions:

- **Nuclear Recoils** (dark matter): Produce short, sharp pulses (rise_time ≈ 2 μs)
- **Electronic Recoils** (background): Produce slower, broader pulses (rise_time ≈ 8 μs)
- **Pile-up**: Overlapping events create combined, distorted waveforms

### Pulse Shape Analysis

The ionization pulse shape reflects the interaction type via:

- **Peak Amplitude**: Total charge collected
- **Rise Time**: Electron drift properties and recoil type (10%-90% timing)
- **Area (Integral)**: Total integrated charge
- **FWHM (Full Width at Half Max)**: Pulse width
- **Chi-Square to Ideal Pulse**: Goodness-of-fit for pile-up detection

These physics-based features enable discrimination without explicit detector simulation.

---

## Solution Overview

This project implements a **two-stage machine learning pipeline**:

1. **Feature Extraction**: Convert raw waveforms into interpretable physics parameters
2. **Classification**: Train a Random Forest to separate signals from background
3. **Robustness Testing**: Validate performance against pile-up and noise

The Random Forest was chosen because it:

- Handles non-linear feature interactions (important for real detector data)
- Provides feature importance rankings (physical insights)
- Generalizes well with limited training data
- Requires no parameter tuning for baseline performance

---

## Methodology

### Step 1: Synthetic Pulse Generation

Generate realistic detector waveforms using physics-based pulse shapes:

```
Signal (Nuclear Recoil):     pulse(t) = A × (t/τ) × exp(-t/τ), τ ≈ 2 μs
Background (Electronic):     pulse(t) = A × (t/τ) × exp(-t/τ), τ ≈ 8 μs
Pile-up:                     pulse(t) = pulse₁(t) + pulse₂(t - Δt)
```

Where:

- `A` = amplitude (peak voltage)
- `τ` = rise time (characteristic decay constant)
- `Δt` = time delay between pile-up events

Gaussian noise added to simulate detector noise (σ ≈ 0.25 V).

### Step 2: Feature Extraction

For each waveform, extract 5 physics-based features:

| Feature        | Description                    | Physics Relevance             |
| -------------- | ------------------------------ | ----------------------------- |
| **Peak**       | Maximum amplitude              | Proportional to recoil energy |
| **Rise Time**  | 10%-90% crossing time          | Distinguishes recoil type     |
| **Area**       | Integral of pulse              | Total ionization produced     |
| **FWHM**       | Full width at half maximum     | Event duration                |
| **Chi-Square** | Goodness-of-fit to ideal pulse | Pile-up detection metric      |

### Step 3: Model Training

- **Data**: 1000 synthetic pulses (500 signal, 500 background)
- **Algorithm**: Random Forest (100 trees)
- **Features**: All 5 physics-based parameters
- **Train/Test Split**: 70%-30%
- **Evaluation**: Classification report, confusion matrix, feature importance

### Step 4: Blind Testing

Generate 50 "mystery" events (mix of signal, background, pile-up) and test model robustness:

- Count false positives (pile-up/noise classified as signal)
- Success = zero false discoveries

---

## Results

### Model Performance

```
Classification Report (Test Set):
                        precision    recall  f1-score   support
      Background (0)       0.86      0.76      0.81       159
      Signal(1)            0.76      0.86      0.81       141

        accuracy                               0.81       300
        macro avg          0.81      0.81      0.81       300
        weighted avg       0.81      0.81      0.81       300

```

### Feature Importance Ranking

1. **Rise Time**: 44% — Most discriminative, directly correlates with recoil type
2. **FWHM**: 10% — Pulse width also differs between event types
3. **Peak**: 12% — Contains some signal information
4. **Area**: 22% — Partially correlated with rise time
5. **Chi-Square**: 13% — Complementary pile-up metric

### Blind Test Results

```
Total mystery events     : 50
Signal candidates found : 0
Background leakage      : 0
STATUS: Discovery SAFE (0 expected background events)
```

---

## Relevance to Physics

### Direct Applications

1. **Dark Matter Searches**: Direct detection experiments rely on robust background rejection. This pipeline demonstrates ML techniques for real detector data processing.

2. **Signal Efficiency**: Achieving high sensitivity (catching real signals) while minimizing background contamination is the holy grail of rare-event searches.

3. **Pile-up Rejection**: Real detectors must handle pile-up. This work shows how physics-informed ML can identify overlapping events.

### Broader Context

- **NEWS-G/DarkSPHERE Collaboration**: Real detectors use similar pulse analysis
- **Generalization**: These techniques apply to other rare-event searches (neutrinos, supernovae, exotic particles)
- **Interpretability**: Physics-based features make ML models transparent and trustworthy

### Why This Matters

Dark matter comprises 85% of matter in the universe, yet remains undetected. Direct detection experiments require unprecedented background rejection. Machine learning, combined with physics domain knowledge, provides the edge needed to distinguish genuine discoveries from instrument artifacts.

---

## Project Structure

```
psa-spc-dm/
├── initialize.py                 # Setup script for directory structure
├── main.py                       # Data generation and feature extraction pipeline
├── requirements.txt              # Project dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
├── data/
│   ├── raw/                      # Original raw waveform data (if applicable)
│   ├── processed/                # Extracted features (CSV)
│   │   └── pulse_features.csv    # Feature dataset (1000 samples × 6 columns)
│   └── simulated/                # Synthetic waveform datasets
│
├── src/                          # Source code
│   ├── __init__.py
│   │
│   ├── simulation/
│   │   └── generate_pulses.py    # Synthetic pulse generation & dataset creation
│   │
│   ├── features/
│   │   └── extraction.py         # Physics-based feature extraction
│   │
│   ├── models/
│   │   ├── train.py              # Random Forest training pipeline
│   │   ├── blind_test.py         # Robustness testing on blind events
│   │   └── evaluate_robustness.py # Performance evaluation metrics
│   │
│   └── utils/
│       └── visualize_separation.py # Feature space visualization
│
├── notebooks/                    # Jupyter notebooks (educational tutorials & analysis)
│   ├── 01_Pulse_Physics_Basics.ipynb          # Learn detector pulse physics
│   ├── 02_Feature_Extraction.ipynb            # Extract physics features from pulses
│   ├── 03_ML_Classification_Training.ipynb    # Train ML classifier
│   └── 04_Robustness_Blind_Testing.ipynb      # Test model on realistic data
│
└── results/
    ├── models/
    │   └── psa_random_forest.pkl  # Trained model weights
    ├── figures/                   # Output plots and visualizations
    └── final_report.py            # Summary and conclusions
```

---

## Setup Instructions

### Requirements

- **Python**: 3.8+
- **OS**: Linux, macOS, Windows
- **Packages**: See `requirements.txt`

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/psa-spc-dm.git
   cd psa-spc-dm
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Initialize project structure** (one-time setup):
   ```bash
   python initialize.py
   ```

### Troubleshooting

- **Import errors**: Ensure virtual environment is activated and dependencies are installed
- **Path errors**: Run commands from the project root directory
- **File not found**: Check that `initialize.py` was executed successfully

---

## Usage Guide

### 1. Generate Data and Extract Features

Generate 1000 synthetic pulses and extract physics-based features:

```bash
python main.py
```

**Output**: `data/processed/pulse_features.csv` (1000 samples with 5 features + labels)

### 2. Train the Model

Train the Random Forest classifier on the extracted features:

```bash
python -m src.models.train
```

**Output**:

- `results/models/psa_random_forest.pkl` (trained model)
- Classification report and feature importance scores

### 3. Run Blind Tests

Test model robustness on 50 unseen "mystery" events:

```bash
python -m src.models.blind_test
```

**Output**: False positive count and robustness assessment

### 4. Complete Pipeline (All Steps)

Run the full pipeline end-to-end:

```bash
python src/final_discovery_run.py
```

This executes:

1. Feature extraction (Step 1)
2. Model training (Step 2)
3. Blind robustness testing (Step 3)
4. Project summary

### 5. Visualize Feature Separation

Create visualizations of feature distributions by signal type:

```bash
python -m src.utils.visualize_separation
```

**Output**: Plots showing feature distributions for signal vs. background

---

## Educational Notebooks

We provide **4 comprehensive Jupyter notebooks** that teach the physics and ML concepts step-by-step in simple language. Each notebook explains the physical relevance of each cell and interprets the results.

### 📙 [Notebook 1: Pulse Physics Basics](notebooks/01_Pulse_Physics_Basics.ipynb)

**What You'll Learn:**

- How detector pulses reflect particle interaction types
- Why nuclear recoils (dark matter) create different pulse shapes than electronic recoils (background)
- What a "pile-up" event looks like and why it's dangerous

**Key Concepts:**

- Physics formula for detector pulses: exponential rise and decay
- Rise time as the primary physical discrimination parameter
- Realistic noise in measurement

**Hands-On:**

- Visualize nuclear vs. electronic recoil pulses side-by-side
- See what pile-up events look like
- Understand why we can tell them apart

---

### 🔬 [Notebook 2: Feature Extraction](notebooks/02_Feature_Extraction.ipynb)

**What You'll Learn:**

- How to extract 5 physics-based features from raw 500-point waveforms
- Why each feature matters physically
- How features separate signal from background

**Key Concepts:**

- **Rise Time**: How fast the pulse grows (KEY discriminator)
- **Peak Amplitude**: Total energy deposited
- **FWHM**: Pulse width and duration
- **Area/Integral**: Total charge collected
- **Chi-Square**: Goodness-of-fit for detecting pile-up

**Hands-On:**

- Understand the math behind each feature calculation
- See example feature values for signal vs. background vs. pile-up
- Visualize how 500 numbers compress to 5 interpretable features

---

### 🤖 [Notebook 3: ML Classification & Training](notebooks/03_ML_Classification_Training.ipynb)

**What You'll Learn:**

- Why machine learning is useful for this problem (non-linear patterns, robustness)
- How Random Forest classifiers work (voting ensemble of decision trees)
- How to train and evaluate an ML model

**Key Concepts:**

- Feature space: how signal and background cluster separately
- Train/test split: avoiding overfitting
- Classification metrics: precision, recall, F1-score
- Feature importance: which features matter most for decisions

**Hands-On:**

- Generate 1000 synthetic training pulses
- Train a 100-tree Random Forest classifier
- Examine confusion matrix and classification report
- See that **Rise Time is 44% of the model's decision** (validates physics!)

---

### 🔬 [Notebook 4: Robustness Testing & Blind Discoveries](notebooks/04_Robustness_Blind_Testing.ipynb)

**What You'll Learn:**

- How to test a trained model under realistic detector conditions
- Why "blind" testing is critical for avoiding false discoveries
- How the model handles pile-up events (the most dangerous background)

**Key Concepts:**

- Blind test: predictions on unseen "mystery" data with unknown composition
- False discovery rate: critical metric for real experiments
- Pile-up rejection: how well the model can identify overlapping events
- ROC curves: understanding model discrimination ability

**Hands-On:**

- Generate 33 mystery events (10 signal + 15 background + 8 pile-up)
- Run trained model without knowing true labels
- Count true positives, false positives, and rejected pile-ups
- **Success metric**: Zero false discoveries = safe for real experiment!

---

### How to Use the Notebooks

1. **Install Jupyter**:

   ```bash
   pip install jupyter
   ```

2. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

3. **Open notebooks**:
   - Navigate to `notebooks/` folder
   - Open `01_Pulse_Physics_Basics.ipynb` and start learning!
   - Run cells sequentially (they build on each other)

4. **Interactive Learning**:
   - Read markdown explanations before each code cell
   - Run code cells to see results
   - Try modifying parameters to see how physics changes!

---

### Why These 4 Notebooks?

| Notebook              | Focus                 | Audience                              |
| --------------------- | --------------------- | ------------------------------------- |
| 1. Pulse Physics      | Physics understanding | Physicists, students new to detectors |
| 2. Feature Extraction | Data engineering      | ML engineers, domain experts          |
| 3. ML Classification  | Machine learning      | Data scientists, ML students          |
| 4. Blind Testing      | Practical validation  | Experimentalists, quality assurance   |

Together, they form a **complete educational story**:

```
Physics → Feature Eng → ML Training → Real-World Testing
  (What)      (How)         (Why)         (Proof)
```

---

## File Descriptions

### Configuration & Setup

- **`initialize.py`**: Creates project directory structure and stub files (run once)
- **`requirements.txt`**: Project dependencies (numpy, pandas, scikit-learn, matplotlib, etc.)
- **`.gitignore`**: Specifies files/folders to exclude from version control

### Main Pipeline

- **`main.py`**: Entry point - orchestrates data generation and feature extraction
- **`src/final_discovery_run.py`**: Runs complete analysis pipeline (data → features → training → testing)

### Simulation Module (`src/simulation/`)

- **`generate_pulses.py`**:
  - `simulate_pulse()`: Generate individual physics-realistic pulses
  - `simulate_pileup()`: Generate overlapping (pile-up) events
  - `generate_dataset()`: Create balanced training dataset (signal + background)

### Feature Extraction (`src/features/`)

- **`extraction.py`**:
  - `extract_features()`: Convert raw waveform → 5 physics features
  - `calculate_chi_square()`: Compute goodness-of-fit metric for pile-up detection
  - `process_batch()`: Batch feature extraction for efficiency

### Machine Learning (`src/models/`)

- **`train.py`**:
  - `train_psa_model()`: Train Random Forest classifier
  - Outputs model weights and evaluation metrics
- **`blind_test.py`**:
  - `run_blind_test()`: Evaluate on 50 unseen events
  - Counts false positives (robustness metric)
- **`evaluate_robustness.py`**: Additional performance analysis and cross-validation

### Utilities (`src/utils/`)

- **`visualize_separation.py`**: Generate visualizations of feature distributions and ROC curves

---

## Contributing

Contributions are welcome! Areas for improvement:

- Feature engineering refinements
- Model optimization and new algorithms
- Documentation and tutorials
- Real detector data support

---

## Contact

**Author**: Samreet Singh Dhillon
**Affiliation**: Panjab University, Chandigarh
**Email**: samreetsinghdhillon@gmail.com

---

## Acknowledgments

This project demonstrates machine learning techniques for rare-event discovery in particle physics. Inspired by real experiments searching for dark matter signatures.
