# ─────────────────────────────────────────────
# 🔧 importlib 패치 (SourceFileLoader, FileFinder)
import importlib._bootstrap as _bootstrap
import importlib.machinery as _machinery

if not hasattr(_bootstrap, 'SourceFileLoader'):
    _bootstrap.SourceFileLoader = _machinery.SourceFileLoader
if not hasattr(_bootstrap, 'FileFinder'):
    _bootstrap.FileFinder = _machinery.FileFinder
# ─────────────────────────────────────────────

import numpy as np
import librosa
import pyworld as pw
import soundfile as sf

# 1) 파일 경로
speech_path = 'output.wav'    # 원본 말하는 음성
song_path   = 'song1.mp3'     # MR 원곡
out_path    = 'karaoke_final.wav'

# 2) 로드 (같은 샘플레이트)
speech, sr = librosa.load(speech_path, sr=None)
song, _    = librosa.load(song_path,    sr=sr)

# 3) MR을 speech 길이에 맞춰 템포 동기화
rate_song = len(speech) / len(song)
song_ts   = librosa.effects.time_stretch(y=song, rate=rate_song)

# 4) WORLD 분석 파라미터
frame_period = 0.3  # ms

# 4a) 보컬(speech) 분석: f0_s, sp, ap
_f0_s, t_s = pw.dio(speech.astype(np.float64), sr, frame_period=frame_period)
f0_s       = pw.stonemask(speech.astype(np.float64), _f0_s, t_s, sr)
sp         = pw.cheaptrick(speech.astype(np.float64), f0_s,       t_s, sr)
ap         = pw.d4c(speech.astype(np.float64),      f0_s,       t_s, sr)

# 4b) MR(song_ts)에서 멜로디 f0_t만 추출
_f0_t, t_t = pw.dio(song_ts.astype(np.float64), sr, frame_period=frame_period)
f0_t       = pw.stonemask(song_ts.astype(np.float64), _f0_t, t_t, sr)

# 5) 보간 & 세미톤 양자화
# 5a) speech 타임라인에 맞춰서 멜로디 보간
f0_lin = np.interp(t_s, t_t, f0_t)
# 5b) 가장 가까운 반음(semitone)으로 양자화
midi = np.round(12 * np.log2(f0_lin / 440.0) + 69)
f0_q = 440.0 * (2 ** ((midi - 69) / 12.0))

# 6) WORLD 합성 → “내 목소리로 노래”
voice = pw.synthesize(f0_q, sp, ap, sr, frame_period=frame_period).astype(np.float32)

# 7) 보컬 + MR 믹스
# 7a) MR 길이 맞추기
if len(song_ts) < len(voice):
    song_ts = np.pad(song_ts, (0, len(voice) - len(song_ts)), 'constant')
else:
    song_ts = song_ts[:len(voice)]

# 7b) 볼륨 밸런스 (MR 작게, 보컬 크게)
mix = voice * 1.0 + song_ts * 0.3

# 7c) 클리핑 방지 정규화
max_amp = np.max(np.abs(mix))
if max_amp > 1.0:
    mix = mix / max_amp

# 8) 결과 저장
sf.write(out_path, mix, sr)
print(f"완료 → {out_path}")
