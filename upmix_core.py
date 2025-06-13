import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly

# 🔹 Filters
def highpass_filter(signal, sr, cutoff_freq):
    sos = butter(4, cutoff_freq, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def lowpass_filter(signal, sr, cutoff_freq):
    sos = butter(4, cutoff_freq, btype='lowpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

# 🔹 Convolution reverb
def apply_reverb_mix(signal, ir, wet_ratio=0.2):
    dry_ratio = 1.0 - wet_ratio
    wet = fftconvolve(signal, ir, mode='full')[:len(signal)]
    return dry_ratio * signal + wet_ratio * wet

# 🔹 Loudness filtering
def loudness_filter(signal, sr):
    sos_hp = butter(2, 50, btype='highpass', fs=sr, output='sos')
    sos_lp = butter(2, 15000, btype='lowpass', fs=sr, output='sos')
    filtered = sosfilt(sos_hp, signal)
    filtered = sosfilt(sos_lp, filtered)
    return filtered

# 🔹 Dolby Atmos-style channel weights
channel_weights = {
0: 1.0,   # Front Left
    1: 1.0,   # Front Right
    2: 0.8,   # Center
    3: 0.5,   # LFE
    4: 0.7,   # Side Left
    5: 0.7,   # Side Right
    6: 0.6,   # Rear Left
    7: 0.6,   # Rear Right
    8: 0.5,   # Top Front Left
    9: 0.5,   # Top Front Right
    10: 0.4,  # Top Rear Left
    11: 0.4   # Top Rear Right
}

# 🔹 True peak detection
def true_peak(signal, oversample=4):
    upsampled = resample_poly(signal, up=oversample, down=1)
    return np.max(np.abs(upsampled))

# 🔹 Loudness and peak calculation
def calculate_atmos_loudness_truepeak(channels, sr):
    weighted_signals = []
    for ch in range(channels.shape[0]):
        weight = channel_weights.get(ch, 1.0)
        filtered = loudness_filter(channels[ch] * weight, sr)
        weighted_signals.append(filtered)

    summed = np.sum(weighted_signals, axis=0)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(summed)
    peak = np.max(np.abs(summed))
    tp = true_peak(summed)
    return loudness, peak, tp

# 🔹 Normalize to target loudness and true peak
def normalize_to_loudness_and_peak(channels, sr, target_loudness=-16.0, max_truepeak=0.995):
    loudness, peak, tp = calculate_atmos_loudness_truepeak(channels, sr)
    print(f"Before normalize: Loudness={loudness:.2f} LUFS, Peak={peak:.3f}, TruePeak={tp:.3f}")

    gain_db = target_loudness - loudness
    gain = 10 ** (gain_db / 20)

    if tp * gain > max_truepeak:
        gain = max_truepeak / tp
        print(f"Adjusted gain to avoid clipping: {gain:.3f}")

    normalized = channels * gain
    loudness, peak, tp = calculate_atmos_loudness_truepeak(normalized, sr)
    print(f"After normalize: Loudness={loudness:.2f} LUFS, Peak={peak:.3f}, TruePeak={tp:.3f}")
    return normalized

# 🔹 Main function
def upmix_and_normalize(y, sr):
    # 🔸 Load IR files (assumed to be in the same directory as the script or repo)
    ir_L_path = "Bricasti M7 Room 02 -Studio B Close-L.wav"
    ir_R_path = "Bricasti M7 Room 02 -Studio B Close-R.wav"
    ir_L, _ = librosa.load(ir_L_path, sr=sr)
    ir_R, _ = librosa.load(ir_R_path, sr=sr)

    # 🔸 Stereo source
    L, R = y[0], y[1]
    mid = (L + R) / 2
    channels = np.zeros((12, len(mid)))

    channels[0] = L * 0.6
    channels[1] = R * 0.6
    channels[2] = mid * 0.1
    channels[3] = lowpass_filter(mid, sr, 80) * 0.15
    channels[4] = apply_reverb_mix(L, ir_L, wet_ratio=0.2) * 0.1
    channels[5] = apply_reverb_mix(R, ir_R, wet_ratio=0.2) * 0.1
    channels[6] = apply_reverb_mix(highpass_filter(L, sr, 100), ir_L, 0.2) * 0.1
    channels[7] = apply_reverb_mix(highpass_filter(R, sr, 100), ir_R, 0.2) * 0.1
    channels[8] = apply_reverb_mix(highpass_filter(L, sr, 150), ir_L, 0.2) * 0.1
    channels[9] = apply_reverb_mix(highpass_filter(R, sr, 150), ir_R, 0.2) * 0.1
    channels[10] = apply_reverb_mix(highpass_filter(L, sr, 200), ir_L, 0.2) * 0.05
    channels[11] = apply_reverb_mix(highpass_filter(R, sr, 200), ir_R, 0.2) * 0.05

    # 🔸 Normalize
    return normalize_to_loudness_and_peak(channels, sr)
