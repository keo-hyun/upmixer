# upmix_logic.py
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly

def highpass_filter(signal, sr, cutoff_freq):
    sos = butter(4, cutoff_freq, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def lowpass_filter(signal, sr, cutoff_freq):
    sos = butter(4, cutoff_freq, btype='lowpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def apply_reverb_mix(signal, ir, wet_ratio=0.2):
    dry_ratio = 1.0 - wet_ratio
    wet = fftconvolve(signal, ir, mode='full')[:len(signal)]
    return dry_ratio * signal + wet_ratio * wet

def loudness_filter(signal, sr):
    sos_hp = butter(2, 50, btype='highpass', fs=sr, output='sos')
    sos_lp = butter(2, 15000, btype='lowpass', fs=sr, output='sos')
    filtered = sosfilt(sos_hp, signal)
    filtered = sosfilt(sos_lp, filtered)
    return filtered

channel_weights = {
    0: 1.0, 1: 1.0, 2: 0.8, 3: 0.5, 4: 0.7, 5: 0.7,
    6: 0.6, 7: 0.6, 8: 0.5, 9: 0.5, 10: 0.4, 11: 0.4
}

def true_peak(signal, oversample=4):
    upsampled = resample_poly(signal, up=oversample, down=1)
    return np.max(np.abs(upsampled))

def calculate_atmos_loudness_truepeak(channels, sr):
    weighted = [loudness_filter(ch * channel_weights.get(i, 1.0), sr) for i, ch in enumerate(channels)]
    summed = np.sum(weighted, axis=0)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(summed)
    peak = np.max(np.abs(summed))
    truepeak = true_peak(summed)
    return loudness, peak, truepeak

def normalize_to_loudness_and_peak(channels, sr, target_loudness=-16.0, max_truepeak=0.995):
    loudness, peak, truepeak = calculate_atmos_loudness_truepeak(channels, sr)
    gain_db = target_loudness - loudness
    gain = 10 ** (gain_db / 20)
    if truepeak * gain > max_truepeak:
        gain = max_truepeak / truepeak
    return channels * gain

def upmix_and_normalize(y, sr, ir_L, ir_R):
    L, R = y[0], y[1]
    mid = (L + R) / 2
    channels = np.zeros((12, len(mid)))

    channels[0] = L * 0.6
    channels[1] = R * 0.6
    channels[2] = mid * 0.1
    channels[3] = lowpass_filter(mid, sr, 80) * 0.15
    channels[4] = apply_reverb_mix(L, ir_L, 0.2) * 0.1
    channels[5] = apply_reverb_mix(R, ir_R, 0.2) * 0.1
    channels[6] = apply_reverb_mix(highpass_filter(L, sr, 100), ir_L, 0.2) * 0.1
    channels[7] = apply_reverb_mix(highpass_filter(R, sr, 100), ir_R, 0.2) * 0.1
    channels[8] = apply_reverb_mix(highpass_filter(L, sr, 150), ir_L, 0.2) * 0.1
    channels[9] = apply_reverb_mix(highpass_filter(R, sr, 150), ir_R, 0.2) * 0.1
    channels[10] = apply_reverb_mix(highpass_filter(L, sr, 200), ir_L, 0.2) * 0.05
    channels[11] = apply_reverb_mix(highpass_filter(R, sr, 200), ir_R, 0.2) * 0.05

    channels_normalized = normalize_to_loudness_and_peak(channels, sr)
    return channels_normalized
