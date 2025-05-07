
import argparse
import subprocess
import shutil
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment, effects

def preclean(input_path, clean_path="clean.wav"):
    raw = AudioSegment.from_file(input_path)
    clean = effects.high_pass_filter(raw, cutoff=120)
    clean = effects.low_pass_filter(clean, cutoff=8000)
    clean = effects.normalize(clean)
    clean.export(clean_path, format="wav")
    return clean_path

def build_timeline(ref_path, sr):
    timeline = []
    if ref_path.lower().endswith('.mid'):
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(ref_path)
        for inst in midi.instruments:
            for note in inst.notes:
                timeline.append((note.start, note.end, note.pitch))
    else:
        y_ref, _ = librosa.load(ref_path, sr=sr)
        _, beat_frames = librosa.beat.beat_track(y=y_ref, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        f0, voiced, _ = librosa.pyin(
            y_ref,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6')
        )
        midi_pitches = np.round(librosa.hz_to_midi(f0))
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
        for i in range(len(beat_times)-1):
            t0, t1 = beat_times[i], beat_times[i+1]
            idx = np.where((times >= t0) & (times < t1))[0]
            if idx.size:
                pitches = midi_pitches[idx]
                valid = pitches[np.isfinite(pitches)]
                if valid.size:
                    target = int(np.median(valid))
                    timeline.append((t0, t1, target))
    return timeline

def get_current_midi(y_seg, sr):
    f0, voiced, _ = librosa.pyin(
        y_seg,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6')
    )
    if np.any(voiced):
        return np.median(librosa.hz_to_midi(f0[voiced]))
    return 60

def split_and_match(y, sr, timeline):
    chunks = []
    for start, end, target in timeline:
        s, e = int(start*sr), int(end*sr)
        dur = end - start
        chunks.append((y[s:e], target, dur))
    return chunks

def tune_segment(seg, sr, target_midi, dur):
    if len(seg)==0:
        return seg
    ratio = (dur*sr)/len(seg)
    stretched = pyrb.time_stretch(seg, sr, ratio)
    current = get_current_midi(stretched, sr)
    n_steps = target_midi - current
    return pyrb.pitch_shift(stretched, sr, n_steps)

def glide(seg, sr, semitones, dur=0.12):
    L = len(seg)
    cut = max(L - int(sr*dur), 0)
    head, tail = seg[:cut], seg[cut:]
    steps = 6
    chunk = max(len(tail)//steps, 1)
    glided = []
    for i in range(steps):
        start, end = i*chunk, (i+1)*chunk
        n = semitones * (i+1)/steps
        glided.append(pyrb.pitch_shift(tail[start:end], sr, n))
    return np.concatenate([head]+glided) if glided else head

def add_vibrato(seg, sr, depth=0.4, rate=5):
    t = np.arange(len(seg))/sr
    lfo = depth*np.sin(2*np.pi*rate*t)
    frame = int(0.04*sr)
    out = []
    for i in range(0, len(seg), frame):
        n = np.mean(lfo[i:i+frame])
        out.append(pyrb.pitch_shift(seg[i:i+frame], sr, n))
    return np.concatenate(out)

def add_expression(y, sr, timeline):
    processed = []
    for seg, target, dur in split_and_match(y, sr, timeline):
        tuned = tune_segment(seg, sr, target, dur)
        semis = target - get_current_midi(tuned, sr)
        gl = glide(tuned, sr, semis)
        vib = add_vibrato(gl, sr)
        processed.append(vib)
    return np.concatenate(processed) if processed else y

def extract_mr(input_path, mr_path="mr.wav"):
    """
    스테레오에서 간단 상쇄 방식으로 MR(반주) 추출
    (좌채널 − 우채널). 모노이면 원본 그대로.
    """
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim==2:
        mr = y[0] - y[1]
    else:
        mr = y
    sf.write(mr_path, mr, sr)
    return mr_path

def apply_fx(input_path, fx_path="sung_fx.wav"):
    """
    • SoX 있으면 reverb+echo → fx_path
    • 없으면 pydub overlay로 간단 echo → fx_path
    """
    if shutil.which("sox"):
        cmd = ["sox", input_path, fx_path, "reverb", "50", "echo", "0.8", "0.9", "220", "0.25"]
        subprocess.run(cmd, check=True)
    else:
        audio = AudioSegment.from_wav(input_path)
        echo = audio - 6
        combined = audio.overlay(echo, position=220)
        combined.export(fx_path, format="wav")
    return fx_path

def mix_with_mr(mr_path, vocal_path, out_path):
    """MR 볼륨 낮춘 뒤 보컬 오버레이 → out_path"""
    mr = AudioSegment.from_wav(mr_path) - 6
    vocal = AudioSegment.from_wav(vocal_path)
    combined = mr.overlay(vocal)
    combined.export(out_path, format="wav")

def make_it_sing(speech_path, melody_ref, out_path):
    # 1) MR 추출
    mr_path = extract_mr(melody_ref, mr_path="mr.wav")
    # 2) 말소리 전처리 + 튠+표현
    clean = preclean(speech_path, clean_path="clean.wav")
    y, sr = librosa.load(clean, sr=None)
    timeline = build_timeline(melody_ref, sr)
    sung = add_expression(y, sr, timeline)
    sf.write("sung.wav", sung, sr)
    # 3) FX 적용
    fx_path = apply_fx("sung.wav", fx_path="sung_fx.wav")
    # 4) MR과 믹싱
    mix_with_mr(mr_path, fx_path, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech를 노래 MR과 합성해 자동 튠 + FX까지 한방에")
    parser.add_argument('--speech', required=True, help='말소리 파일(wav/mp3)')
    parser.add_argument('--melody', required=True, help='노래 파일(wav 또는 midi)')
    parser.add_argument('--out', default='final_mix.wav', help='최종 출력 WAV')
    args = parser.parse_args()
    make_it_sing(args.speech, args.melody, args.out)
