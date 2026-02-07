import numpy as np

def calculate_chi_square(t, pulse, peak_val, rise_time):
    # Add a safety check for zero rise_time
    if rise_time <= 0:
        return 999.0 # Assign a high error value for bad pulses
    
    t0 = t[np.argmax(pulse)] - (rise_time * 0.5)
    ideal = np.zeros_like(t)
    mask = t > t0
    
    # Standard SPC pulse shape formula
    ideal[mask] = peak_val * ((t[mask] - t0) / rise_time) * np.exp(-(t[mask] - t0) / rise_time)
    
    residuals = np.sum((pulse - ideal)**2) / len(t)
    return residuals

def extract_features(t, pulse):
    """Distills a single waveform into physical parameters."""
    # 1. Baseline subtraction (using first 15 samples)
    baseline = np.mean(pulse[:15])
    clean_pulse = pulse - baseline
    
    # 2. Peak Amplitude
    peak_val = np.max(clean_pulse)
    if peak_val <= 0: return [0, 0, 0, 0] 
    
    # 3. Rise Time (10% to 90% of peak)
    threshold_10 = 0.10 * peak_val
    threshold_90 = 0.90 * peak_val
    
    # Find first indices crossing thresholds
    try:
        idx_10 = np.where(clean_pulse >= threshold_10)[0][0]
        idx_90 = np.where(clean_pulse >= threshold_90)[0][0]
        rise_time = t[idx_90] - t[idx_10]
    except IndexError:
        rise_time = 0

    # 4. Total Area (Pulse Integral = Total Charge)
    area = np.trapezoid(clean_pulse, t)
    
    # 5. Pulse Width (FWHM)
    half_max = peak_val / 2.0
    over_half = np.where(clean_pulse >= half_max)[0]
    fwhm = t[over_half[-1]] - t[over_half[0]] if len(over_half) > 0 else 0
    # 6. Chi-Square to Ideal Pulse (for pile-up detection)
    chi_sq = calculate_chi_square(t, clean_pulse, peak_val, rise_time)
    return [peak_val, rise_time, area, fwhm, chi_sq]

def process_batch(t, pulses):
    """Applies extraction to an entire array of pulses."""
    features = [extract_features(t, p) for p in pulses]
    return np.array(features)
