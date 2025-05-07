
import numpy as np
import librosa
import soundfile as sf
import time

# 측정 시작
start = time.time()

# --- 1) 파일 로드 및 상수 정의 ---
speech_path       = 'output.wav'
vocals_path       = 'only_voice4.mp3'
inst_path         = 'only_music4.mp3'
out_path          = 'final_test.wav'

speech, sr        = librosa.load(speech_path, sr=None)
vocals, _         = librosa.load(vocals_path, sr=sr)
instrumental, _   = librosa.load(inst_path, sr=sr)

# 파라미터
frame_length = 2048
hop_length   = 512
fmin, fmax   = librosa.note_to_hz('C2'), librosa.note_to_hz('C7')

# --- 2) 노래 박자 감지 ---
_, beat_frames = librosa.beat.beat_track(y=vocals, sr=sr, hop_length=hop_length)
beat_samples_v = librosa.frames_to_samples(beat_frames, hop_length=hop_length)
beat_samples_v = np.concatenate([beat_samples_v, [len(vocals)]])
durations_v    = np.diff(beat_samples_v)
n_blocks       = len(durations_v)

# --- 3) 무음 판단 (RMS) ---
rms_blocks = np.array([
    np.sqrt(np.mean(vocals[s:e]**2))
    for s, e in zip(beat_samples_v[:-1], beat_samples_v[1:])
])
silence_thr = np.median(rms_blocks) * 0.3
is_silent   = rms_blocks < silence_thr

# --- 4) instrumental 연장 ---
output_len = durations_v.sum()
inst_ext = np.tile(instrumental, int(np.ceil(output_len / len(instrumental))))[:output_len]

# --- 5) 순차 매핑 & 블록 단위 튜닝+믹스 ---
speech_ptr = 0
inst_ptr   = 0
mixed_blocks = []
for i, dur_v in enumerate(durations_v):
    # instrumental block
    inst_blk = inst_ext[inst_ptr:inst_ptr+dur_v]
    inst_ptr += dur_v

    if is_silent[i]:
        tuned = np.zeros(dur_v, dtype=np.float32)
    else:
        # 원본 speech 조각
        seg = speech[speech_ptr:speech_ptr+dur_v]
        speech_ptr += dur_v

        # 시간 비율 계산 (speech len vs target)
        dur_s = len(seg)
        if dur_s == 0:
            stretched = np.zeros(dur_v)
        else:
            rate = dur_s / dur_v
            # 속도 변화 제한 (0.8~1.2)
            rate = np.clip(rate, 0.8, 1.2)
            try:
                stretched = librosa.effects.time_stretch(seg, rate=rate)
            except Exception:
                stretched = seg
        # pad/trim
        if len(stretched) < dur_v:
            stretched = np.pad(stretched, (0, dur_v - len(stretched)), mode='constant')
        else:
            stretched = stretched[:dur_v]

        # pitch-shift
        v_seg = vocals[beat_samples_v[i]:beat_samples_v[i+1]]
        # median f0
        pad_len = max(len(stretched), len(v_seg), frame_length)
        s_pad = np.pad(stretched, (0, pad_len - len(stretched)), mode='constant')
        v_pad = np.pad(v_seg, (0, pad_len - len(v_seg)), mode='constant')
        f0_v = np.median(librosa.yin(v_pad,  fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=frame_length, hop_length=hop_length))
        f0_s = np.median(librosa.yin(s_pad, fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=frame_length, hop_length=hop_length))
        n_steps = 12 * np.log2(f0_v / f0_s) if f0_v>0 and f0_s>0 else 0.0
        shifted = librosa.effects.pitch_shift(y=s_pad, sr=sr, n_steps=n_steps)
        tuned = shifted[:dur_v]
        if len(tuned) < dur_v:
            tuned = np.pad(tuned, (0, dur_v - len(tuned)), mode='constant')

    mixed_blocks.append(tuned + 0.5 * inst_blk)

# --- 6) 블록 연결 & 출력 ---
out = np.concatenate(mixed_blocks)
out /= np.max(np.abs(out)) + 1e-8
sf.write(out_path, out, sr)

end = time.time()
print(f"✅ 저장: '{out_path}' | 길이: {len(out)/sr:.2f}초 | 실행 시간: {end-start:.2f}초")

