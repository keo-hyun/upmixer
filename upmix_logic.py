import logging
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, oaconvolve, resample_poly

# ---------------- 필터 및 유틸 ----------------
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

# ---------------- True Peak 정규화 ----------------
def normalize_truepeak(channels, oversample=2, max_tp=0.891):
    """
    다채널 신호를 true peak 기준으로 정규화
    """
    mix = np.max(np.abs(channels), axis=0)  # peak envelope
    upsampled = resample_poly(mix, up=oversample, down=1)
    tp = np.max(np.abs(upsampled))
    gain = max_tp / tp if tp > 0 else 1.0
    return channels * gain

# ---------------- 업믹스 ----------------
def upmix_and_normalize(y, sr, ir_L, ir_R, output_format="7.1.4"):
    L, R = y[0], y[1]
    mid = (L + R) / 2
    channels = np.zeros((12, len(mid)))

    # Dolby/ITU 채널 순서 기준
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

    # 출력 포맷별 채널 선택
    format_channel_map = {
        "5.1":     [0, 1, 2, 3, 4, 5],
        "5.1.2":   [0, 1, 2, 3, 4, 5, 8, 9],
        "7.1":     [0, 1, 2, 3, 4, 5, 6, 7],
        "7.1.2":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "7.1.4":   list(range(12))
    }

    selected_indices = format_channel_map.get(output_format, list(range(12)))
    selected_channels = channels[selected_indices]

    return normalize_truepeak(selected_channels)

# ---------------- 전체 실행 ----------------
def upmix(input_path, output_path, ir_L, ir_R, ir_sr, output_format="7.1.4"):
    logging.info(f"[UPMIX] 시작 - input: {input_path}, format: {output_format}")
    
    # load
    y, sr = librosa.load(input_path, sr=None, mono=False, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input file must be stereo")

    # resample IR
    if ir_sr != sr:
        logging.info(f"[UPMIX] IR resample: {ir_sr} → {sr}")
        ir_L = librosa.resample(ir_L.astype(np.float32), ir_sr, sr)
        ir_R = librosa.resample(ir_R.astype(np.float32), ir_sr, sr)

    # process
    upmixed = upmix_and_normalize(y, sr, ir_L, ir_R, output_format=output_format)

    # save
    sf.write(output_path, upmixed.T, sr)
    logging.info(f"[UPMIX] 완료 - output saved to: {output_path}")
