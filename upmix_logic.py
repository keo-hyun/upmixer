import numpy as np
import librosa
import soundfile as sf
import subprocess
import os
from scipy.signal import butter, sosfilt, oaconvolve, resample_poly
import pyloudnorm as pyln

# ---------------- 필터 및 유틸 ----------------
def highpass_filter(signal, sr, cutoff):
    sos = butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def lowpass_filter(signal, sr, cutoff):
    sos = butter(4, cutoff, btype='lowpass', fs=sr, output='sos')
    return sosfilt(sos, signal)

def apply_reverb_mix(signal, ir, wet=0.2):
    dry = 1.0 - wet
    wet_signal = oaconvolve(signal, ir, mode='full', method='fft')[:len(signal)]
    return dry * signal + wet * wet_signal

# ---------------- Loudness 정규화 ----------------
def true_peak(signal, oversample=2):
    upsampled = resample_poly(signal, up=oversample, down=1)
    return np.max(np.abs(upsampled))

def calculate_truepeak(channels):
    stacked = np.stack(channels, axis=0)
    peak = np.max(np.abs(stacked), axis=0)
    return true_peak(peak)

def normalize_by_truepeak(channels, sr, max_tp=0.891):
    tp = calculate_truepeak(channels)
    gain = max_tp / tp if tp > 0 else 1.0
    return channels * gain

# ---------------- FFmpeg 처리 ----------------
def get_ffmpeg_layout(fmt):
    return {
        "5.1": "5.1",
        "5.1.2": "5.1.2",
        "7.1": "7.1",
        "7.1.2": "7.1.2",
        "7.1.4": "7.1.4"
    }.get(fmt, "7.1.4")

def write_with_ffmpeg(temp_path, output_path, layout="7.1.4"):
    cmd = [
        "ffmpeg", "-y", "-i", temp_path,
        "-channel_layout", layout,
        "-c:a", "pcm_s24le", output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(temp_path)

# ---------------- 업믹스 처리 ----------------
def upmix_and_normalize(y, sr, ir_L, ir_R, output_format="7.1.4"):
    L, R = y[0], y[1]
    mid = (L + R) / 2
    channels = np.zeros((12, len(mid)))

    # Dolby/ITU 표준 채널 순서
    channels[0] = L * 0.6                                # L
    channels[1] = R * 0.6                                # R
    channels[2] = mid * 0.1                              # C
    channels[3] = lowpass_filter(mid, sr, 80) * 0.15     # LFE
    channels[4] = apply_reverb_mix(L, ir_L, 0.2) * 0.1   # LS
    channels[5] = apply_reverb_mix(R, ir_R, 0.2) * 0.1   # RS
    channels[6] = apply_reverb_mix(highpass_filter(L, sr, 100), ir_L, 0.2) * 0.1  # LB
    channels[7] = apply_reverb_mix(highpass_filter(R, sr, 100), ir_R, 0.2) * 0.1  # RB
    channels[8] = apply_reverb_mix(highpass_filter(L, sr, 150), ir_L, 0.2) * 0.1  # TFL
    channels[9] = apply_reverb_mix(highpass_filter(R, sr, 150), ir_R, 0.2) * 0.1  # TFR
    channels[10] = apply_reverb_mix(highpass_filter(L, sr, 200), ir_L, 0.2) * 0.05  # TBL
    channels[11] = apply_reverb_mix(highpass_filter(R, sr, 200), ir_R, 0.2) * 0.05  # TBR

    # -------- Dolby 호환 교치 로직 (LS/RS ↔ LB/RB) --------
    channels[[4, 5, 6, 7]] = channels[[6, 7, 4, 5]]  # swap LS<->LB, RS<->RB

    # 포맷별 채널 선택
    format_map = {
        "5.1":     [0, 1, 2, 3, 4, 5],
        "5.1.2":   [0, 1, 2, 3, 4, 5, 8, 9],
        "7.1":     [0, 1, 2, 3, 4, 5, 6, 7],
        "7.1.2":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "7.1.4":   list(range(12))
    }

    indices = format_map.get(output_format, list(range(12)))
    selected = channels[indices]

    return normalize_by_truepeak(selected, sr)

# ---------------- 전체 처리 함수 ----------------
def upmix(input_path, output_path, ir_L, ir_R, ir_sr, output_format="7.1.4"):
    y, sr = librosa.load(input_path, sr=None, mono=False, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input must be stereo WAV")

    # IR 리샘플링
    if ir_sr != sr:
        ir_L = librosa.resample(ir_L.astype(np.float32), ir_sr, sr)
        ir_R = librosa.resample(ir_R.astype(np.float32), ir_sr, sr)

    upmixed = upmix_and_normalize(y, sr, ir_L, ir_R, output_format)

    # WAV 저장 + FFmpeg 레이아웃 처리
    temp_path = output_path.replace(".wav", "_temp.wav")
    sf.write(temp_path, upmixed.T, sr)
    write_with_ffmpeg(temp_path, output_path, get_ffmpeg_layout(output_format))
