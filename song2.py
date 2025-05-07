
import argparse
import subprocess
import shutil
import sys

import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment, effects

def extract_mr(input_path, sr=None, mr_path="mr.wav"):
    """
    스테레오에서 좌−우 채널 상쇄로 MR 추출.
    Mono 입력이면 경고 후 원본을 MR로 사용합니다.
    """
    y, orig_sr = librosa.load(input_path, sr=sr, mono=False)
    if y.ndim != 2:
        print("[WARNING] MR 추출: 입력 파일이 스테레오가 아닙니다. 원본을 MR로 사용합니다.", file=sys.stderr)
        sf.write(mr_path, y, orig_sr)
        return mr_path, orig_sr
    mr = y[0] - y[1]
    sf.write(mr_path, mr, orig_sr)
    return mr_path, orig_sr

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
                    timeline.append((t0, t1, int(np.median(valid))))
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

def tune_and_express(y, sr, timeline):
    segments = []
    for start, end, target in timeline:
        s, e = int(start*sr), int(end*sr)
        seg = y[s:e]
        if seg.size == 0:
            continue
        # 1) time-stretch + pitch-shift
        ratio = (end - start) * sr / seg.size
        stretched = pyrb.time_stretch(seg, sr, ratio)
        current = get_current_midi(stretched, sr)
        tuned = pyrb.pitch_shift(stretched, sr, target - current)
        # 2) glide
        cut = max(len(tuned) - int(sr*0.12), 0)
        head, tail = tuned[:cut], tuned[cut:]
        steps = 6
        chunk = max(len(tail)//steps, 1)
        glided = [
            pyrb.pitch_shift(tail[i*chunk:(i+1)*chunk], sr,
                             (target - current)*(i+1)/steps)
            for i in range(steps)
        ]
        merged = np.concatenate([head, *glided]) if glided else tuned
        # 3) vibrato
        t = np.arange(len(merged))/sr
        lfo = 0.4 * np.sin(2*np.pi*5*t)
        frame = int(0.04*sr)
        vib = []
        for i in range(0, len(merged), frame):
            n = np.mean(lfo[i:i+frame])
            vib.append(pyrb.pitch_shift(merged[i:i+frame], sr, n))
        final_seg = np.concatenate(vib)
        segments.append((final_seg, start))
    return segments

def apply_fx_to_wave(in_path, out_path):
    if shutil.which("sox"):
        cmd = ["sox", in_path, out_path, "reverb", "50", "echo", "0.8", "0.9", "220", "0.25"]
        subprocess.run(cmd, check=True)
    else:
        audio = AudioSegment.from_wav(in_path)
        echo = audio - 6
        combined = audio.overlay(echo, position=220)
        combined.export(out_path, format="wav")

def make_it_sing(speech_path, melody_path, out_path):
    # 1) MR 추출 (mono fallback 포함)
    mr_path, mr_sr = extract_mr(melody_path)
    mr_seg, _ = sf.read(mr_path)
    # 2) speech 전처리
    clean = preclean(speech_path)
    y, sr = librosa.load(clean, sr=None)
    # 3) 타임라인 만들기
    timeline = build_timeline(melody_path, sr)
    # 4) 튠+표현 → 세그먼트 리스트
    voiced = tune_and_express(y, sr, timeline)
    # 5) MR 길이에 맞춰 voice_data 배열에 쌓기
    voice_data = np.zeros_like(mr_seg, dtype=np.float32)
    for seg, start in voiced:
        s = int(start * sr)
        e = s + len(seg)
        voice_data[s:e] += seg[:len(voice_data[s:e])]
    sf.write("voice_raw.wav", voice_data, sr)
    # 6) FX 적용
    apply_fx_to_wave("voice_raw.wav", "voice_fx.wav")
    # 7) MR(−6dB) 위에 보컬 오버레이
    mr_audio    = AudioSegment.from_wav(mr_path) - 6
    vocal_audio = AudioSegment.from_wav("voice_fx.wav")
    final       = mr_audio.overlay(vocal_audio)
    final.export(out_path, format="wav")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--speech', required=True, help='말소리 파일(wav)')
    p.add_argument('--melody', required=True, help='노래 파일 (wav 또는 midi)')
    p.add_argument('--out',    default='final_mix.wav', help='최종 출력 wav')
    args = p.parse_args()
    make_it_sing(args.speech, args.melody, args.out)
