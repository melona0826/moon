
import argparse
import subprocess
import shutil
import sys

import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment, effects

def extract_mr(input_path, mr_path="mr.wav"):
    """
    AudioSegment.from_file() 으로 MP3/WAV 읽기.
    • 스테레오면 좌−우 채널 상쇄 → MR
    • 모노면 경고 후 원본을 MR로 사용
    """
    audio = AudioSegment.from_file(input_path)
    channels = audio.split_to_mono()
    if len(channels) == 2:
        left, right = channels
        mr_seg = left.overlay(right.invert_phase())
    else:
        print("[WARNING] Mono input for MR. Using original track as MR.", file=sys.stderr)
        mr_seg = channels[0]
    mr_seg.export(mr_path, format="wav")
    mr_array, sr = sf.read(mr_path)
    return mr_path, mr_array, sr

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
                valid = midi_pitches[idx][np.isfinite(midi_pitches[idx])]
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
            pyrb.pitch_shift(
                tail[i*chunk:(i+1)*chunk], sr,
                (target - current)*(i+1)/steps
            ) for i in range(steps)
        ]
        merged = np.concatenate([head, *glided]) if glided else tuned
        # 3) vibrato
        t = np.arange(len(merged)) / sr
        lfo = 0.4 * np.sin(2 * np.pi * 5 * t)
        frame = int(0.04 * sr)
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
    # 1) MR 추출 (MP3/WAV, stereo/mono 모두 지원)
    mr_path, mr_seg, sr = extract_mr(melody_path)
    # 2) speech 전처리 & matching sample rate
    clean = preclean(speech_path)
    y, _ = librosa.load(clean, sr=sr)
    # 3) timeline 생성
    timeline = build_timeline(melody_path, sr)
    # 4) 튠+표현 → segment 리스트
    voiced = tune_and_express(y, sr, timeline)
    # 5) MR 길이에 맞춰 voice_data 생성
    voice_data = np.zeros_like(mr_seg, dtype=np.float32)
    for seg, start in voiced:
        s = int(start * sr)
        e = s + len(seg)
        voice_data[s:e] += seg[:len(voice_data[s:e])]
    sf.write("voice_raw.wav", voice_data, sr)
    # 6) FX 적용
    apply_fx_to_wave("voice_raw.wav", "voice_fx.wav")
    # 7) MR을 튠된 보컬 길이에 맞춰 자르기
    vocal_audio = AudioSegment.from_wav("voice_fx.wav")
    mr_audio    = AudioSegment.from_file(mr_path) - 6
    mr_cropped  = mr_audio[:len(vocal_audio)]
    # 8) 오버레이 및 최종 출력
    final = mr_cropped.overlay(vocal_audio)
    final.export(out_path, format="wav")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--speech', required=True, help='말소리 파일(wav/mp3)')
    p.add_argument('--melody', required=True, help='노래 파일(wav/mp3 또는 midi)')
    p.add_argument('--out',    default='final_mix.wav', help='최종 출력 WAV')
    args = p.parse_args()
    make_it_sing(args.speech, args.melody, args.out)
