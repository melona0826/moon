
import argparse
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
import aubio

# ─── 스케일 정의 ────────────────────────────────────────────────
SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
}
NOTE_TO_SEMITONE = {
    'C':  0,  'C#': 1,  'Db': 1,
    'D':  2,  'D#': 3,  'Eb': 3,
    'E':  4,
    'F':  5,  'F#': 6,  'Gb': 6,
    'G':  7,  'G#': 8,  'Ab': 8,
    'A':  9,  'A#':10,  'Bb':10,
    'B': 11
}

def nearest_scale_shift(midi_note, key_root, intervals):
    rel = (midi_note - key_root) % 12
    cands = intervals + [i+12 for i in intervals]
    diffs = [cand - rel for cand in cands]
    return min(diffs, key=lambda x: abs(x))

# ─── 프레임별 보정 ────────────────────────────────────────────────
def compute_shifts(y, sr, hop, win, key_root, intervals):
    """
    aubio의 pitch 객체를 이용해 프레임별 MIDI 피치를 검출합니다.
    aubio.pitch()에는 hop_size 길이만큼 샘플을 입력해야 하므로,
    hop 길이의 프레임을 전달하도록 수정했습니다.
    """
    pitch_o = aubio.pitch("yinfft", win, hop, sr)
    pitch_o.set_unit("midi")
    pitch_o.set_silence(-40)

    n_frames = int(np.ceil(len(y) / hop))
    shifts = np.zeros(n_frames, dtype=float)
    for i in range(n_frames):
        start = i * hop
        frame = y[start:start + hop]
        if len(frame) < hop:
            frame = np.pad(frame, (0, hop - len(frame)))
        m = pitch_o(frame.astype('float32'))[0]
        shifts[i] = 0.0 if m <= 0 else nearest_scale_shift(m, key_root, intervals)
    return shifts

# ─── Overlap-Add 방식 보정 ──────────────────────────────────────────
def apply_ola_pitchshift(y, sr, shifts, hop, win):
    n_frames = len(shifts)
    out_len = len(y) + win
    out = np.zeros(out_len, dtype=float)
    norm = np.zeros(out_len, dtype=float)
    window = np.hanning(win)

    for idx, n_steps in enumerate(shifts):
        start = idx * hop
        frame = y[start:start + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        frame_win = frame * window
        # pitch_shift 호출 시 키워드 인자를 사용하여 파라미터 매칭 오류 방지
        shifted = librosa.effects.pitch_shift(y=frame_win, sr=sr, n_steps=n_steps)
        if len(shifted) < win:
            shifted = np.pad(shifted, (0, win - len(shifted)))
        else:
            shifted = shifted[:win]
        out[start:start + win] += shifted * window
        norm[start:start + win] += window**2

    nonzero = norm > 1e-6
    out[nonzero] /= norm[nonzero]
    return out[:len(y)]

# ─── Auto-Tune 처리 함수 ───────────────────────────────────────────
def autotune_wav(in_wav, out_wav, key, scale, hop, win):
    y, sr = librosa.load(in_wav, sr=None)
    key_root = NOTE_TO_SEMITONE[key]
    intervals = SCALE_INTERVALS[scale]
    shifts = compute_shifts(y, sr, hop, win, key_root, intervals)
    y_tuned = apply_ola_pitchshift(y, sr, shifts, hop, win)
    sf.write(out_wav, y_tuned, sr)
    print(f"✅ Auto-tuned WAV saved to {out_wav}")

# ─── 백그라운드 오버레이 ─────────────────────────────────────────
def overlay_with_song(vocals_wav, song_mp3, output_mp3, vocals_gain, song_gain):
    vocals = AudioSegment.from_file(vocals_wav, format="wav")
    song   = AudioSegment.from_file(song_mp3, format="mp3")
    if len(song) < len(vocals):
        reps = len(vocals)//len(song) + 1
        song = (song * reps)[:len(vocals)]
    else:
        song = song[:len(vocals)]
    vocals = vocals.apply_gain(vocals_gain)
    song   = song.apply_gain(song_gain)
    out = song.overlay(vocals)
    out.export(output_mp3, format="mp3")
    print(f"✅ Final songified MP3 saved to {output_mp3}")

# ─── 커맨드라인 인터페이스 ─────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Python으로 Real Auto-Tune Meme 생성")
    p.add_argument('voice_wav', help="입력 음성 WAV")
    p.add_argument('song_mp3',  help="배경 노래 MP3")
    p.add_argument('--key',     choices=NOTE_TO_SEMITONE.keys(), default='C',
                   help="키 (예: C, D#, F)")
    p.add_argument('--scale',   choices=SCALE_INTERVALS.keys(), default='major',
                   help="스케일 (major/minor)")
    p.add_argument('--hop',     type=int, default=512,  help="hop size (samples)")
    p.add_argument('--win',     type=int, default=2048, help="window size (samples)")
    p.add_argument('--vocals-gain', type=float, default=0.0, help="보컬 볼륨(dB)")
    p.add_argument('--song-gain',   type=float, default=-6.0, help="노래 볼륨(dB)")
    p.add_argument('--tuned-wav',   default='tuned.wav',  help="중간 WAV 파일명")
    p.add_argument('--output',      default='songified.mp3', help="결과 MP3 파일명")
    args = p.parse_args()

    autotune_wav(
        args.voice_wav, args.tuned_wav,
        key=args.key, scale=args.scale,
        hop=args.hop, win=args.win
    )
    overlay_with_song(
        args.tuned_wav,
        args.song_mp3,
        args.output,
        vocals_gain=args.vocals_gain,
        song_gain=args.song_gain
    )

if __name__ == '__main__':
    main()
