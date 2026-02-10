# 📚 How to Get Started with the Notebooks

## Quick Start for Learning

If you want to **understand this project from first principles**, follow the notebooks in order:

### Step 1: Install Jupyter (if not already installed)

```bash
pip install jupyter
```

### Step 2: Open Jupyter in the notebooks folder

```bash
cd notebooks
jupyter notebook
```

### Step 3: Open and run notebooks in this order:

#### **First: Notebook 1 - Pulse Physics Basics** ⭐ START HERE

- **File**: `01_Pulse_Physics_Basics.ipynb`
- **Duration**: ~10 minutes to read + run
- **What you'll learn**:
  - How detector pulses look different for dark matter vs background
  - Why rise time is the key discriminator
  - What pile-up events are and why they're dangerous
- **Run all cells** from top to bottom

---

#### **Second: Notebook 2 - Feature Extraction**

- **File**: `02_Feature_Extraction.ipynb`
- **Duration**: ~15 minutes
- **What you'll learn**:
  - How to extract 5 physics-based features from raw pulses
  - Why each feature matters
  - How 500-point waveforms become 5 numbers
- **Prerequisites**: Notebook 1 knowledge

---

#### **Third: Notebook 3 - ML Classification & Training**

- **File**: `03_ML_Classification_Training.ipynb`
- **Duration**: ~20 minutes
- **What you'll learn**:
  - How machine learning separates signal from background
  - Why Rise Time is 44% of the model's decision (matches physics!)
  - How to evaluate classifier performance
- **Prerequisites**: Notebooks 1 & 2

---

#### **Fourth: Notebook 4 - Robustness & Blind Testing**

- **File**: `04_Robustness_Blind_Testing.ipynb`
- **Duration**: ~15 minutes
- **What you'll learn**:
  - How to test a trained model on realistic data
  - Why pile-up detection is critical (avoid false discoveries)
  - How to measure model robustness
- **Prerequisites**: All previous notebooks

---

## Learning Paths by Background

### For Physics Students/Professors

→ Start with **Notebook 1** (pulse physics)  
→ Quick skim of **Notebook 2** (just understand the idea)  
→ Focus on **Notebooks 3 & 4** (ML applications, real-world validation)

### For ML/Data Science Students

→ Read **Notebook 1** briefly (understand the physics context)  
→ Deep dive into **Notebooks 2 & 3** (feature engineering + ML)  
→ **Notebook 4** (robustness testing, preventing false discoveries)

### For Experimentalists/Engineers

→ All notebooks in order (understand full pipeline)  
→ Pay special attention to **Notebook 4** (how to avoid false discoveries)

---

## Key Learning Objectives

By completing all 4 notebooks, you'll understand:

✅ **Physics**: Why dark matter and background pulses look different  
✅ **Engineering**: How to extract meaningful features from raw detector data  
✅ **Machine Learning**: How to train classifiers for rare-event detection  
✅ **Validation**: Why blind testing and false discovery control are critical

---

## Pro Tips

1. **Run cells one at a time** - Don't use "Run All"
2. **Read the markdown explanations** - They explain physics and math before each code cell
3. **Modify the code** - Try changing parameters to see how physics changes
4. **Take notes** - The key insights are in the explanations, not just the plots
5. **Ask questions** - Physics-informed ML is best understood by thinking critically

---

## Troubleshooting

**Q: I get an import error when running a notebook**  
A: Make sure you're in the correct conda/venv environment and ran `pip install -r requirements.txt` from the project root

**Q: The plots look different from the ones in the notebook descriptions**  
A: That's expected! Notebooks use random seeds for reproducibility, but your data may vary. The patterns should be similar.

**Q: Can I skip a notebook?**  
A: Not recommended. Notebook 2 uses concepts from 1, Notebook 3 uses 2, etc. They build on each other.

**Q: How long does it take to run all notebooks?**  
A: ~60 minutes total (10+15+20+15 minutes, plus thinking time)

---

## What's Next?

After completing the notebooks, you can:

1. **Run the full pipeline**: `python src/final_discovery_run.py`
2. **Read the main source code** in `src/` (now that you understand the concepts)
3. **Modify the project**: Try different ML algorithms, adjust feature extraction, etc.
4. **Experiment with real data**: Adapt the code to your own detector readout

---

Happy learning! 🔬✨
