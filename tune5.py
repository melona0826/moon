
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
    """
    midi_note: 실수형 미디 노트 (예: 60.3)
    key_root: 0~11 정수 (C=0, C#=1 …)
    intervals: 스케일 내에서 허용하는 0~11 범위의 반음 간격 리스트
    """
    # 해당 옥타브 내 상대 위치
    rel = (midi_note - key_root) % 12
    # 후보 = intervals, intervals+12
    cands = intervals + [i+12 for i in intervals]
    # rel 위치에서 가장 가까운 cand를 찾아 차이(semis) 리턴
    diffs = [cand - rel for cand in cands]
    semis = min(diffs, key=lambda x: abs(x))
    return semis

# ─── 프레임별 보정 ────────────────────────────────────────────────
def autotune_wav(in_wav, out_wav, key='C', scale='major',
                 hop_s=512, win_s=2048, sr=None):
    """
    in_wav  : 입력 음성 WAV
    out_wav : 출력 보정 WAV
    key     : 'C', 'D#', ...
    scale   : 'major' or 'minor'
    hop_s   : aubio hop size
    win_s   : aubio window size
    sr      : None 이면 파일에 저장된 sr 사용
    """
    # 1) aubio 피치 추적기 세팅 (YIN)
    samplerate = sr or 0
    pitch_o    = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_silence(-40)

    # 2) 원본 로드
    y, fs = librosa.load(in_wav, sr=sr)
    # 3) 프레임별 pitch 추출
    midi_pitches = []
    for i in range(0, len(y), hop_s):
        frame = y[i:i+win_s]
        if len(frame) < win_s:
            frame = np.pad(frame, (0, win_s - len(frame)))
        m = pitch_o(frame.astype('float32'))[0]
        midi_pitches.append(m if m > 0 else np.nan)

    # 4) 각 프레임별 보정 semitone 계산
    key_root = NOTE_TO_SEMITONE[key]
    intervals = SCALE_INTERVALS[scale]
    shifts = []
    for m in midi_pitches:
        if np.isnan(m):
            shifts.append(0.0)
        else:
            shifts.append(nearest_scale_shift(m, key_root, intervals))

    # 5) 프레임 단위로 pitch_shift 적용 후 overlap-add (단순 non-overlap)
    out_audio = np.zeros_like(y)
    idx = 0
    for shift_semitones in shifts:
        frame = y[idx:idx+win_s]
        # librosa는 mono numpy array
        shifted = librosa.effects.pitch_shift(frame, fs, n_steps=shift_semitones)
        length = min(len(shifted), len(out_audio)-idx)
        out_audio[idx:idx+length] = shifted[:length]
        idx += hop_s

    # 6) 파일로 저장
    sf.write(out_wav, out_audio, fs)
    print(f"✅ Auto-tuned WAV saved to {out_wav}")

# ─── 백그라운드 오버레이 ─────────────────────────────────────────
def overlay_with_song(vocals_wav, song_mp3, output_mp3,
                      vocals_gain_dB=0.0, song_gain_dB=-6.0):
    vocals = AudioSegment.from_file(vocals_wav, format="wav")
    song   = AudioSegment.from_file(song_mp3, format="mp3")

    # 길이 맞추기
    if len(song) < len(vocals):
        reps = len(vocals)//len(song) + 1
        song = (song*reps)[:len(vocals)]
    else:
        song = song[:len(vocals)]

    vocals = vocals.apply_gain(vocals_gain_dB)
    song   = song.apply_gain(song_gain_dB)

    out = song.overlay(vocals)
    out.export(output_mp3, format="mp3")
    print(f"✅ Final songified MP3 saved to {output_mp3}")

# ─── 커맨드라인 인터페이스 ─────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Python Auto-Tune Meme Generator"
    )
    p.add_argument('voice_wav',  help="입력 음성 WAV 파일")
    p.add_argument('song_mp3',   help="백그라운드 노래 MP3 파일")
    p.add_argument('--key',      default='C',
                   choices=NOTE_TO_SEMITONE.keys(),
                   help="튜닝할 키 (예: C, D#, F)")
    p.add_argument('--scale',    default='major',
                   choices=SCALE_INTERVALS.keys(),
                   help="스케일 (major/minor)")
    p.add_argument('--hop',      type=int, default=512,
                   help="피치 분석 hop size")
    p.add_argument('--win',      type=int, default=2048,
                   help="피치 분석 window size")
    p.add_argument('--vocals-gain', type=float, default=0.0,
                   help="보컬 볼륨 조정 (dB)")
    p.add_argument('--song-gain',   type=float, default=-6.0,
                   help="노래 볼륨 조정 (dB)")
    p.add_argument('--tuned-wav',   default='tuned.wav',
                   help="중간 보정 WAV 파일명")
    p.add_argument('--output',      default='songified.mp3',
                   help="최종 MP3 파일명")
    args = p.parse_args()

    autotune_wav(
        args.voice_wav,
        args.tuned_wav,
        key=args.key,
        scale=args.scale,
        hop_s=args.hop,
        win_s=args.win
    )
    overlay_with_song(
        args.tuned_wav,
        args.song_mp3,
        args.output,
        vocals_gain_dB=args.vocals_gain,
        song_gain_dB=args.song_gain
    )

if __name__ == '__main__':
    main()
