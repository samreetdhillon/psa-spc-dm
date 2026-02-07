from src.simulation.generate_pulses import generate_dataset
from src.features.extraction import process_batch
import pandas as pd

# 1. Generate Data
print("Generating synthetic pulses...")
t, X, y = generate_dataset(n_samples=500) # 500 signal, 500 background

# 2. Extract Features
print("Extracting features...")
features = process_batch(t, X)

# 3. Save to Dataframe
df = pd.DataFrame(features, columns=['peak', 'rise_time', 'area', 'fwhm', 'chi_sq'])
df['label'] = y

df.to_csv("data/processed/pulse_features.csv", index=False)
print("Saved 1000 extracted features to data/processed/pulse_features.csv")

# Quick Look
print(df.groupby('label')[['rise_time', 'fwhm']].mean())