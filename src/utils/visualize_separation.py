import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_separation():
    # Load the features we just extracted
    df = pd.read_csv("data/processed/pulse_features.csv")
    
    plt.figure(figsize=(10, 6))
    
    # We plot Rise Time vs Peak (Energy proxy)
    # This is the "Golden Plot" in NEWS-G PSA
    sns.scatterplot(data=df, x='peak', y='rise_time', hue='label', alpha=0.5, palette='viridis')
    
    plt.title("Background Discrimination: Rise Time vs. Peak Amplitude")
    plt.xlabel("Peak Amplitude (Energy Proxy)")
    plt.ylabel("Rise Time (μs)")
    plt.legend(title='Event Type', labels=['Background (ER)', 'Signal (NR)'])
    
    plt.savefig("results/figures/separation_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_separation()