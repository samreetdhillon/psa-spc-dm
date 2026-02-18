# PSA-SPC-DM

_A machine learning pipeline for discriminating dark matter signals from background noise in Spherical Proportional Counter (SPC) detectors using pulse shape analysis (PSA) and physics-based feature extraction._

## Problem

The search for Dark Matter is a "needle in a haystack" problem. Dark matter detection experiments like NEWS-G and DarkSPHERE use Spherical Proportional Counter (SPC), in which a Dark Matter interaction (Signal) produces a tiny electronic pulse. Unfortunately, natural radioactivity and electronic noise (Background) produce pulses that look almost identical.

So the challenge is to distinguish a rare Dark Matter ping from the constant static of the universe. If we cannot perfectly filter out the background, we risk a False Discovery, claiming we found Dark Matter when we actually just measured detector noise.

## Solution

Pulse Shape Analysis (PSA) leverages the unique physics of electron drift in a sphere.

- **The Signal (Nuclear Recoil):** Dark Matter hits a single point, creating a tight cluster of electrons that arrive at the sensor nearly all at once. This creates a **sharp, fast pulse**.
- **The Background (Electronic Recoil):** Radioactive particles leave long, streaking tracks. Electrons arrive at the sensor at different times, creating a **wide, sluggish pulse**.

By measuring the **Rise Time** and the "physicality" (**Chi-Square**) of these pulses, we can create a digital filter to separate the ghosts from the noise.

## Methodology

I built a pipeline in Python to test this discrimination strategy:

1.  **Physics-Based Simulation:** Generated synthetic pulses using response functions that mimic real SPC electronics, including Gaussian noise and "pile-up" (overlapping events).
2.  **Feature Extraction:** Processed raw waveforms into five physical descriptors:
    - _Peak Amplitude_ (Energy proxy)
    - _Rise Time_ (10% to 90% speed)
    - _FWHM_ (Pulse width)
    - _Area_ (Total collected charge)
    - _Chi-Square ($\chi^2$)_ (Template matching to catch lumpy, non-physical pulses).
3.  **Machine Learning:** Trained a **Random Forest Classifier** to learn the difference between Signal and Background.
4.  **The Discovery Cut:** Implemented an ultra-conservative 99.5% probability threshold to ensure zero background leakage.

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

## How to Run

1. **Generate & Extract Features:** `python main.py`
2. **Train Model:** `python src/models/train.py`
3. **Run Blind Tests:** `python -m src.models.blind_test`
4. **Analyze Physics:** `python src/utils/physical_analysis.py`
5. **Visualize Feature Separation:** `python -m src.utils.visualize_separation`
6. **Final Run:** `python src/final_discovery_run.py`

## Contact

Samreet Singh Dhillon

Panjab University, Chandigarh

samreetsinghdhillon@gmail.com
