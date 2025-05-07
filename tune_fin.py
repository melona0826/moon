import numpy as np
import librosa
import soundfile as sf
import time

start = time.time()

# --- 1) 파일 로드 및 길이 패딩 함수 정의 ---
def match_length(x, L):
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)), mode='constant')
    else:
        x = x[:L]
    return x

speech, sr       = librosa.load('output.wav',       sr=None)
vocals, _        = librosa.load('only_voice.mp3',   sr=sr)
instrumental, _  = librosa.load('only_music.mp3',   sr=sr)

# --- 2) vocals 기준으로 speech, instrumental 패딩 ---
L_vocals = len(vocals)
speech       = match_length(speech,      L_vocals)
instrumental = match_length(instrumental,L_vocals)

# --- 3) 박자(비트) 감지 on vocals ---
tempo, beat_frames = librosa.beat.beat_track(y=vocals, sr=sr, hop_length=512)
beat_samples       = librosa.frames_to_samples(beat_frames, hop_length=512)
beat_samples       = np.concatenate([beat_samples, [L_vocals]])  # 마지막 구간

# --- 4) 블록별 무음 판단 (RMS) ---
n_blocks     = len(beat_samples) - 1
rms_blocks   = np.array([
    np.sqrt(np.mean(vocals[s:e]**2))
    for s, e in zip(beat_samples[:-1], beat_samples[1:])
])
silence_thr  = np.median(rms_blocks) * 0.3
is_silent    = rms_blocks < silence_thr

# --- 5) 각 블록별 튜닝 & 믹스 ---
fmin, fmax = librosa.note_to_hz('C2'), librosa.note_to_hz('C7')
mixed_blocks = []

for i in range(n_blocks):
    s, e = beat_samples[i], beat_samples[i+1]
    dur  = e - s

    # 반주 블록 자르기
    inst_blk = instrumental[s:e]

    if is_silent[i]:
        # 무음 블록: voice = 0
        tuned_blk = np.zeros(dur, dtype=np.float32)
    else:
        # vocals + speech 블록
        v = vocals[s:e]
        sp = speech[s:e]
        # 최소 frame_length 확보를 위해 패딩
        min_len = max(len(v), len(sp), 2048)
        v_pad  = np.pad(v,  (0, min_len - len(v)),  mode='constant')
        sp_pad = np.pad(sp, (0, min_len - len(sp)), mode='constant')

        # 블록 하나당 median f0 계산
        f0_v = np.median(librosa.yin(v_pad,  fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=2048, hop_length=512))
        f0_s = np.median(librosa.yin(sp_pad, fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=2048, hop_length=512))
        # semitone shift
        if f0_v > 0 and f0_s > 0:
            n_steps = 12 * np.log2(f0_v / f0_s)
        else:
            n_steps = 0.0

        # pitch shift 후 원 길이로 자르기
        shifted = librosa.effects.pitch_shift(y=sp_pad, sr=sr, n_steps=n_steps)
        tuned_blk = shifted[:dur]
        if len(tuned_blk) < dur:
            tuned_blk = np.pad(tuned_blk, (0, dur - len(tuned_blk)))

    # voice + 50% 반주 믹스
    mixed = tuned_blk + 0.5 * inst_blk
    mixed_blocks.append(mixed)

# --- 6) 블록 연결 & 저장 ---
mixed_audio = np.concatenate(mixed_blocks)
mixed_audio /= np.max(np.abs(mixed_audio)) + 1e-8

sf.write('final_2min.wav', mixed_audio, sr)
print(f"✅ 최종 결과물이 길이 {len(mixed_audio)/sr:.2f}초로 저장되었습니다.")

end = time.time()
print(f"실행 시간: {end - start:.6f}초")
