import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_separation():
    # Load the extracted features
    df = pd.read_csv("data/processed/pulse_features.csv")
    
    plt.figure(figsize=(10, 6))
    
    # Plot Rise Time vs Peak (Energy proxy)
    sns.scatterplot(data=df, x='peak', y='rise_time', hue='label', alpha=0.5, palette='viridis')
    
    plt.title("Background Discrimination: Rise Time vs. Peak Amplitude")
    plt.xlabel("Peak Amplitude (Energy Proxy)")
    plt.ylabel("Rise Time (μs)")
    plt.legend(title='Event Type', labels=['Background (ER)', 'Signal (NR)'])
    
    plt.savefig("results/figures/separation_plot.png")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import joblib

def plot_probability_distribution():
    df = pd.read_csv("data/processed/pulse_features.csv")
    model = joblib.load("results/models/psa_random_forest.pkl")
    
    # Get probabilities
    probs = model.predict_proba(df[['peak', 'rise_time', 'area', 'fwhm', 'chi_sq']])[:, 1]
    df['prob'] = probs
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['label']==0]['prob'], bins=50, alpha=0.5, label='Background (ER)', color='orange')
    plt.hist(df[df['label']==1]['prob'], bins=50, alpha=0.5, label='Signal (NR)', color='blue')
    
    # Draw the Discovery Cut
    plt.axvline(0.995, color='red', linestyle='--', label='Discovery Cut (99.5%)')
    
    plt.title("Model Confidence Distribution")
    plt.xlabel("Probability of being Signal")
    plt.ylabel("Event Count")
    plt.yscale('log') # Log scale helps see small leakage
    plt.legend()
    plt.savefig("results/figures/probability_dist.png")
    plt.show()

def plot_feature_importance():
    # Load model and define feature names
    model = joblib.load("results/models/psa_random_forest.pkl")
    features = ['Peak', 'Rise Time', 'Area', 'FWHM', 'Chi-Square']
    importances = model.feature_importances_

    # Create the plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(features))
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    
    plt.barh(y_pos, importances[sorted_idx], color='skyblue', edgecolor='navy')
    plt.yticks(y_pos, [features[i] for i in sorted_idx])
    plt.xlabel('Importance Score (Relative Weight)')
    plt.title('Physical Feature Significance in Pulse Shape Analysis')
    
    # Add labels on bars
    for i, v in enumerate(importances[sorted_idx]):
        plt.text(v + 0.01, i, f'{v:.2%}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/figures/feature_importance.png")
    plt.show()

if __name__ == "__main__":
    plot_separation()
    plot_probability_distribution()
    plot_feature_importance()