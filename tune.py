
"""
Match speech pitch and tempo to a piano song using aubio for analysis and sox for transformation.
"""

import subprocess
import numpy as np
from aubio import source, tempo, pitch

def estimate_tempo(path):
    win_s, hop_s = 1024, 512
    src = source(path, 0, hop_s)
    o = tempo("default", win_s, hop_s, src.samplerate)
    beats = []

    while True:
        samples, read = src()
        if o(samples):
            beats.append(o.get_last_s())
        if read < hop_s:
            break

    if len(beats) > 1:
        bpms = 60.0 / np.diff(beats)
        return np.median(bpms)
    else:
        raise RuntimeError(f"Could not detect beats in {path}")

def estimate_pitch(path):
    win_s, hop_s = 2048, 1024
    src = source(path, 0, hop_s)
    p = pitch("yin", win_s, hop_s, src.samplerate)
    p.set_unit("Hz")
    p.set_silence(-40)
    pitches = []

    while True:
        samples, read = src()
        pitch_est = p(samples)[0]
        if pitch_est > 0:
            pitches.append(pitch_est)
        if read < hop_s:
            break

    if pitches:
        return np.array(pitches)
    else:
        raise RuntimeError(f"Could not detect pitch in {path}")

def match_tempo_and_pitch(speech_path, song_path, output_path):
    # 1) 템포 측정
    t_speech = estimate_tempo(speech_path)
    t_song   = estimate_tempo(song_path)
    rate = t_song / t_speech
    print(f"Speech tempo: {t_speech:.2f} BPM, Song tempo: {t_song:.2f} BPM, rate = {rate:.3f}")

    # 2) 피치 측정
    p_speech = estimate_pitch(speech_path)
    p_song   = estimate_pitch(song_path)
    median_speech = np.median(p_speech)
    max_song = np.max(p_song)
    semitones = 12 * np.log2(max_song / median_speech)
    print(f"Median speech pitch: {median_speech:.1f} Hz, Max song pitch: {max_song:.1f} Hz, shift = {semitones:.2f} semitones")

    # 3) sox로 변환 적용 (tempo → pitch)
    cents = semitones * 100  # sox는 센트 단위
    cmd = [
        "sox", speech_path, output_path,
        "tempo", f"{rate:.4f}",
        "pitch", f"{cents:.1f}"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved matched audio to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Match speech's pitch & tempo to a piano song")
    parser.add_argument("--speech", default="output.wav", help="Input speech WAV file")
    parser.add_argument("--song",   default="song1.mp3", help="Reference piano MP3 file")
    parser.add_argument("--output", default="matched_output.wav", help="Transformed output file")
    args = parser.parse_args()
    match_tempo_and_pitch(args.speech, args.song, args.output)
