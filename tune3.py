#!/usr/bin/env python3
import subprocess
from pydub import AudioSegment

def apply_autotune(input_wav: str,
                   output_wav: str,
                   autotalent_so: str = '/usr/lib/ladspa/autotalent.so',
                   key: str = 'C',
                   scale: str = 'major',
                   shift: float = 0.0,
                   amount: float = 100.0,
                   mix: float = 100.0):
    """
    FFmpeg + LADSPA Autotalent으로 음성 WAV에 오토튠 적용
    - input_wav: 입력 음성 파일(.wav)
    - output_wav: LADSPA가 WAV만 처리하므로 중간 WAV 파일
    - key, scale: 튜닝 키와 스케일
    - shift: cent 단위 피치 이동, amount: 튜닝 강도(%), mix: dry/wet 비율(%)
    """
    ladspa_args = (
        f"file={autotalent_so}:plugin=autotalent:"
        f"shift={shift}:amount={amount}:mix={mix}:"
        f"in_key={key}:scale={scale}"
    )
    cmd = [
        'ffmpeg', '-y', '-i', input_wav,
        '-af', f"ladspa={ladspa_args}",
        output_wav
    ]
    subprocess.run(cmd, check=True)

def overlay_with_song(vocals_wav: str,
                      song_mp3: str,
                      output_mp3: str,
                      vocals_gain_dB: float = 0.0,
                      song_gain_dB: float = -6.0):
    """
    튠된 음성을 노래 MP3 위에 오버레이
    - vocals_wav: 튠된 음성 파일(.wav)
    - song_mp3: 배경이 될 노래 파일(.mp3)
    - output_mp3: 결과물(.mp3)
    - 볼륨 조절용 gain 값
    """
    vocals = AudioSegment.from_file(vocals_wav)
    song   = AudioSegment.from_file(song_mp3)
    # 둘 길이 맞추기: 음성이 짧으면 노래를 음성 길이에 맞추고,
    # 음성이 길면 노래를 반복해서 음성 길이를 채움
    if len(song) < len(vocals):
        repeats = len(vocals) // len(song) + 1
        song = (song * repeats)[:len(vocals)]
    else:
        song = song[:len(vocals)]
    vocals = vocals.apply_gain(vocals_gain_dB)
    song   = song.apply_gain(song_gain_dB)
    combined = song.overlay(vocals, position=0)
    combined.export(output_mp3, format='mp3')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="WAV 음성 + MP3 노래로 Songify Meme 생성")
    parser.add_argument('voice_wav',
                        help="튜닝할 음성 WAV 파일 경로")
    parser.add_argument('song_mp3',
                        help="배경으로 사용할 노래 MP3 파일 경로")
    parser.add_argument('--key',     default='C',
                        help="튜닝 키 (예: C, D#, F)")
    parser.add_argument('--scale',   default='major',
                        help="스케일 (major/minor)")
    parser.add_argument('--shift',   type=float, default=0.0,
                        help="전체 피치 이동(cent 단위)")
    parser.add_argument('--amount',  type=float, default=100.0,
                        help="튜닝 강도 (%)")
    parser.add_argument('--mix',     type=float, default=100.0,
                        help="Dry/Wet 믹스 (%)")
    parser.add_argument('--autotalent', default='/usr/lib/ladspa/autotalent.so',
                        help="Autotalent LADSPA 플러그인 경로")
    parser.add_argument('--output',  default='songified.mp3',
                        help="출력 MP3 파일명")
    args = parser.parse_args()

    # 1) 음성 WAV에 Auto-Tune 적용 (중간 WAV 생성)
    apply_autotune(
        args.voice_wav,
        'tuned_voice.wav',
        autotalent_so=args.autotalent,
        key=args.key,
        scale=args.scale,
        shift=args.shift,
        amount=args.amount,
        mix=args.mix
    )

    # 2) 튠된 음성과 노래 MP3 합성
    overlay_with_song(
        'tuned_voice.wav',
        args.song_mp3,
        args.output
    )

    print(f"✅ 완성! 생성된 파일: {args.output}")
