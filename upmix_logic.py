import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import subprocess
import os
from scipy.signal import butter, sosfilt, oaconvolve, resample_poly

def highpass_filter(signal, sr, cutoff_freq):
    sos = butter(4, cutoff_freq, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def lowpass_filter(signal, sr, cutoff_freq):
    sos = butter(4, cutoff_freq, btype='lowpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def apply_reverb_mix(signal, ir, wet_ratio=0.2):
    dry_ratio = 1.0 - wet_ratio
    wet = oaconvolve(signal, ir, mode='full', method='fft')[:len(signal)]
    return dry_ratio * signal + wet_ratio * wet

def true_peak(signal, oversample=2):
    upsampled = resample_poly(signal, up=oversample, down=1)
    return np.max(np.abs(upsampled))

def calculate_truepeak_only(channels, sr):
    flattened = np.max(np.abs(np.stack(channels, axis=0)), axis=0)
    return true_peak(flattened)

def normalize_by_truepeak(channels, sr, max_truepeak=0.891):
    tp = calculate_truepeak_only(channels, sr)
    gain = max_truepeak / tp if tp > 0 else 1.0
    return channels * gain

def get_ffmpeg_channel_layout(format_str):
    layout_map = {
        "5.1":     "5.1",
        "5.1.2":   "5.1.2",
        "7.1":     "7.1",
        "7.1.2":   "7.1.2",
        "7.1.4":   "7.1.4"
    }
    return layout_map.get(format_str, "7.1.4")

def write_with_ffmpeg(temp_path, final_path, layout="7.1.4"):
    cmd = [
        "ffmpeg", "-y", "-i", temp_path,
        "-channel_layout", layout,
        "-c:a", "pcm_s24le", final_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(temp_path)

def upmix_and_normalize(y, sr, ir_L, ir_R, output_format="7.1.4"):
    L, R = y[0], y[1]
    mid = (L + R) / 2
    channels = np.zeros((12, len(mid)))

    # Dolby/ITU 기준 채널 배치
    channels[0] = L * 0.6  # L
    channels[1] = R * 0.6  # R
    channels[2] = mid * 0.1  # C
    channels[3] = lowpass_filter(mid, sr, 80) * 0.15  # LFE
    channels[4] = apply_reverb_mix(L, ir_L, 0.2) * 0.1  # LS
    channels[5] = apply_reverb_mix(R, ir_R, 0.2) * 0.1  # RS
    channels[6] = apply_reverb_mix(highpass_filter(L, sr, 100), ir_L, 0.2) * 0.1  # LB
    channels[7] = apply_reverb_mix(highpass_filter(R, sr, 100), ir_R, 0.2) * 0.1  # RB
    channels[8] = apply_reverb_mix(highpass_filter(L, sr, 150), ir_L, 0.2) * 0.1  # TFL
    channels[9] = apply_reverb_mix(highpass_filter(R, sr, 150), ir_R, 0.2) * 0.1  # TFR
    channels[10] = apply_reverb_mix(highpass_filter(L, sr, 200), ir_L, 0.2) * 0.05  # TBL
    channels[11] = apply_reverb_mix(highpass_filter(R, sr, 200), ir_R, 0.2) * 0.05  # TBR

    format_channel_map = {
        "5.1":     [0, 1, 2, 3, 4, 5],
        "5.1.2":   [0, 1, 2, 3, 4, 5, 8, 9],
        "7.1":     [0, 1, 2, 3, 4, 5, 6, 7],
        "7.1.2":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "7.1.4":   list(range(12))
    }

    selected_indices = format_channel_map.get(output_format, list(range(12)))
    selected_channels = channels[selected_indices]

    normalized = normalize_by_truepeak(selected_channels, sr)
    return normalized

def upmix(input_path: str, output_path: str, ir_L: np.ndarray, ir_R: np.ndarray, ir_sr: int, output_format: str = "7.1.4"):
    y, sr = librosa.load(input_path, sr=None, mono=False, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input file must be a stereo WAV file.")

    if ir_sr != sr:
        ir_L = librosa.resample(ir_L.astype(np.float32), orig_sr=ir_sr, target_sr=sr)
        ir_R = librosa.resample(ir_R.astype(np.float32), orig_sr=ir_sr, target_sr=sr)

    upmixed = upmix_and_normalize(y, sr, ir_L, ir_R, output_format=output_format)

    temp_path = output_path.replace(".wav", "_temp.wav")
    sf.write(temp_path, upmixed.T, sr)

    layout = get_ffmpeg_channel_layout(output_format)
    write_with_ffmpeg(temp_path, output_path, layout)
