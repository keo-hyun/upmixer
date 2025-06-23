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
    sos_hp = butter(2, 20, btype='highpass', fs=sr, output='sos')
    sos_lp = butter(2, 20000, btype='lowpass', fs=sr, output='sos')
    return sosfilt(sos_lp, sosfilt(sos_hp, signal))

declare_channel_weights = lambda: {
    0: 1.0, 1: 1.0, 2: 0.8, 3: 0.5, 4: 0.7, 5: 0.7,
    6: 0.6, 7: 0.6, 8: 0.5, 9: 0.5, 10: 0.4, 11: 0.4
}
channel_weights = declare_channel_weights()

def true_peak(signal, oversample=4):
    upsampled = resample_poly(signal, up=oversample, down=1)
    return np.max(np.abs(upsampled))

def calculate_truepeak_only(channels, sr):
    flattened = np.max(np.abs(np.stack(channels, axis=0)), axis=0)
    return true_peak(flattened)

def calculate_lufs_truepeak(channels, sr):
    flattened = np.sqrt(np.mean(np.square(np.array(channels)), axis=0))
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(flattened)
    tp = true_peak(flattened)
    return lufs, tp

def normalize_by_truepeak(channels, sr, max_truepeak=0.891):
    tp = calculate_truepeak_only(channels, sr)
    gain = max_truepeak / tp if tp > 0 else 1.0
    return channels * gain

def upmix_and_normalize(y, sr, ir_L, ir_R, output_format="7.1.4"):
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

    format_channel_map = {
        "5.1":     list(range(6)),
        "5.1.2":   list(range(6)) + [8, 9],
        "7.1":     list(range(8)),
        "7.1.2":   list(range(10)),
        "7.1.4":   list(range(12))
    }
    selected_indices = format_channel_map.get(output_format, list(range(12)))
    selected_channels = channels[selected_indices]

    normalized = normalize_by_truepeak(selected_channels, sr)
    return normalized

def upmix(input_path: str, output_path: str, ir_L: np.ndarray, ir_R: np.ndarray, ir_sr: int, output_format: str = "7.1.4"):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input file must be a stereo WAV file.")

    if sr != ir_sr:
        ir_L = librosa.resample(ir_L, orig_sr=ir_sr, target_sr=sr)
        ir_R = librosa.resample(ir_R, orig_sr=ir_sr, target_sr=sr)
        ir_sr = sr

    upmixed = upmix_and_normalize(y, sr, ir_L, ir_R, output_format=output_format)
    sf.write(output_path, upmixed.T, sr)
