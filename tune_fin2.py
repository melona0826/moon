import numpy as np
import librosa
import soundfile as sf
import time

start = time.time()

speech_path       = 'output2.wav'
vocals_path       = 'only_voice5.mp3'
inst_path         = 'only_music5.mp3'
out_path          = 'tune_fin2.wav'

speech, sr        = librosa.load(speech_path,      sr=None)
vocals, _         = librosa.load(vocals_path,      sr=sr)
instrumental, _   = librosa.load(inst_path,        sr=sr)

frame_length = 2048
hop_length   = 512
fmin, fmax   = librosa.note_to_hz('C2'), librosa.note_to_hz('C7')

_, beat_frames = librosa.beat.beat_track(y=vocals, sr=sr, hop_length=hop_length)
beat_samples   = librosa.frames_to_samples(beat_frames, hop_length=hop_length)
beat_samples   = np.concatenate([beat_samples, [len(vocals)]])


durations = np.diff(beat_samples)
n_blocks  = len(durations)
rms_blocks = np.array([
    np.sqrt(np.mean(vocals[s:e]**2))
    for s, e in zip(beat_samples[:-1], beat_samples[1:])
])
silence_thr = np.median(rms_blocks) * 0.5
is_silent   = rms_blocks < silence_thr

# --- 4) instrumental 연장 ---
output_len = durations.sum()
inst_ext   = np.tile(instrumental, int(np.ceil(output_len / len(instrumental))))
inst_ext   = inst_ext[:output_len]

speech_ptr = 0
inst_ptr   = 0
mixed_blocks = []

for i, dur in enumerate(durations):
    inst_blk = inst_ext[inst_ptr:inst_ptr+dur]
    inst_ptr += dur

    if is_silent[i]:
        tuned = np.zeros(dur, dtype=np.float32)
    else:
        seg = speech[speech_ptr:speech_ptr+dur]
        speech_ptr += dur
        v_seg = vocals[beat_samples[i]:beat_samples[i+1]]

        min_len = max(len(seg), len(v_seg), frame_length)
        seg_pad = np.pad(seg,   (0, min_len - len(seg)), mode='constant')
        v_pad   = np.pad(v_seg, (0, min_len - len(v_seg)), mode='constant')

        f0_v = np.median(librosa.yin(v_pad,  fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=frame_length, hop_length=hop_length))
        f0_s = np.median(librosa.yin(seg_pad, fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=frame_length, hop_length=hop_length))
        n_steps = 12 * np.log2(f0_v / f0_s) if f0_v>0 and f0_s>0 else 0.0

        shifted = librosa.effects.pitch_shift(y=seg_pad, sr=sr, n_steps=n_steps)
        tuned = shifted[:dur]
        if len(tuned) < dur:
            tuned = np.pad(tuned, (0, dur - len(tuned)), mode='constant')

    mixed_blocks.append(tuned + 0.5 * inst_blk)


out = np.concatenate(mixed_blocks)


# trim_seconds = 50.0
# trim_samples = int(trim_seconds * sr)
# out = out[:trim_samples]


out /= np.max(np.abs(out)) + 1e-8
sf.write(out_path, out, sr)

print(f"✅ 최종 출력을 '{out_path}'에 저장했습니다.")
end = time.time()
print(f"실행 시간: {end - start:.6f}초")
