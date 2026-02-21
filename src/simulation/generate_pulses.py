import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_pulse(t, t0, amplitude, rise_time, noise_level=0.40):
    """
    Generates a synthetic SPC pulse using a simplified response function.
    t: time array
    t0: start time of pulse
    amplitude: peak of the pulse
    rise_time: governs the width/slope of the pulse
    """
    pulse = np.zeros_like(t)
    mask = t > t0
    # Physical approximation: Rise governed by electron drift, 
    # decay governed by pre-amp RC constant
    pulse[mask] = amplitude * ((t[mask] - t0) / rise_time) * np.exp(-(t[mask] - t0) / rise_time)
    
    # Add Gaussian Noise
    noise = np.random.normal(0, noise_level, size=len(t))
    return pulse + noise

def generate_dataset(n_samples=100):
    t = np.linspace(0, 100, 500) # 500 samples over 100 microseconds
    data = []
    labels = []

    for _ in range(n_samples):
        # Signal (Nuclear Recoil): Sharp rise, short duration
        sig_pulse = simulate_pulse(t, t0=20, amplitude=1.0, rise_time=2.0)
        data.append(sig_pulse)
        labels.append(1) # 1 for Signal

        # Background (Electronic Recoil): Slower rise, more spread out
        bg_pulse = simulate_pulse(t, t0=20, amplitude=0.8, rise_time=8.0)
        data.append(bg_pulse)
        labels.append(0) # 0 for Background

    return t, np.array(data), np.array(labels)

def simulate_pileup(t, t0, noise_level=0.25):
    """Simulates two background events hitting close together."""
    # First background event
    p1 = simulate_pulse(t, t0=t0, amplitude=0.6, rise_time=8.0, noise_level=0)
    # Second background event slightly delayed
    p2 = simulate_pulse(t, t0=t0+15, amplitude=0.5, rise_time=7.0, noise_level=0)
    
    noise = np.random.normal(0, noise_level, size=len(t))
    return p1 + p2 + noise

if __name__ == "__main__":
    t, X, y = generate_dataset(n_samples=1)
    plt.plot(t, X[0], label="Signal (Nuclear)")
    plt.plot(t, X[1], label="Background (Electronic)")
    plt.title("Synthetic SPC Pulses")
    plt.legend()
    plt.show()

