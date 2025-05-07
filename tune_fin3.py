import numpy as np
import librosa
import soundfile as sf
import time

start = time.time()

speech_path       = 'output2.wav'
vocals_path       = 'only_voice5.mp3'
inst_path         = 'only_music5.mp3'
out_path          = 'tune_fin3.wav'

speech, sr      = librosa.load(speech_path, sr=None)
vocals, _       = librosa.load(vocals_path, sr=sr)
instrumental, _ = librosa.load(inst_path, sr=sr)

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

output_len = durations.sum()
inst_ext   = np.tile(instrumental, int(np.ceil(output_len / len(instrumental))))[:output_len]

speech_ptr = 0
inst_ptr   = 0
mixed_blocks = []

def is_block_silent(v_seg, thresh):
    rms  = np.sqrt(np.mean(v_seg**2))
    peak = np.max(np.abs(v_seg))
    return rms < thresh and peak < (thresh * 0.5)

for i, dur in enumerate(durations):
    inst_blk = inst_ext[inst_ptr:inst_ptr+dur]
    inst_ptr += dur

    s_v, e_v = beat_samples[i], beat_samples[i+1]
    v_seg    = vocals[s_v:e_v]

    if is_block_silent(v_seg, silence_thr):
        tuned = np.zeros(dur, dtype=np.float32)
    else:
        seg = speech[speech_ptr:speech_ptr+dur]
        speech_ptr += dur

        pad_len = max(len(seg), len(v_seg), frame_length)
        seg_pad = np.pad(seg,   (0, pad_len - len(seg)), mode='constant')
        v_pad   = np.pad(v_seg, (0, pad_len - len(v_seg)), mode='constant')

        f0_v = np.median(librosa.yin(v_pad, fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=frame_length, hop_length=hop_length))
        f0_s = np.median(librosa.yin(seg_pad, fmin=fmin, fmax=fmax,
                                     sr=sr, frame_length=frame_length, hop_length=hop_length))

        n_steps = 12 * np.log2(f0_v / f0_s) if f0_v>0 and f0_s>0 else 0.0

        f_tgt = f0_s * (2 ** (n_steps / 12))

        f_tgt_clamped = np.clip(f_tgt, 20.0, sr / 2 - 100.0)

        n_steps_clamped = 12 * np.log2(f_tgt_clamped / f0_s) if f0_s>0 else 0.0


        shifted = librosa.effects.pitch_shift(y=seg_pad, sr=sr, n_steps=n_steps_clamped)
        tuned = shifted[:dur]
        if len(tuned) < dur:
            tuned = np.pad(tuned, (0, dur - len(tuned)), mode='constant')

    mixed_blocks.append(tuned + 0.5 * inst_blk)


out = np.concatenate(mixed_blocks)
out /= np.max(np.abs(out)) + 1e-8
sf.write(out_path, out, sr)

end = time.time()
print(f"Output Name :  '{out_path}. Time: {end-start:.2f}s, Length: {len(out)/sr:.2f}s")

