
import subprocess
import argparse
from pydub import AudioSegment

def apply_pitch_sox(input_wav: str,
                    output_wav: str,
                    cents: float = 100.0):
    """
    SoX의 pitch 이펙트로 센트 단위 피치 쉬프트
    - input_wav: 입력 음성 WAV 파일 경로
    - output_wav: 출력 WAV 파일 경로
    - cents: 올릴(+) / 내릴(-) cent 값 (100 cents = 1 semitone)
    """
    cmd = [
        'sox',
        input_wav,
        output_wav,
        'pitch',
        str(cents)
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Pitch shifted by {cents} cents → {output_wav}")

def overlay_with_song(vocals_wav: str,
                      song_mp3: str,
                      output_mp3: str,
                      vocals_gain_dB: float = 0.0,
                      song_gain_dB: float = -6.0):
    """
    튜닝된 음성을 배경 노래 위에 오버레이하여 MP3로 출력
    - vocals_wav: 피치 쉬프트된 음성 WAV 파일
    - song_mp3: 배경으로 사용할 노래 MP3 파일
    - output_mp3: 최종 출력 MP3 파일명
    - vocals_gain_dB: 음성 볼륨 조정 (dB)
    - song_gain_dB: 노래 볼륨 조정 (dB)
    """
    vocals = AudioSegment.from_file(vocals_wav, format="wav")
    song   = AudioSegment.from_file(song_mp3, format="mp3")

    # 길이 맞추기: 노래가 짧으면 반복, 길면 자름
    if len(song) < len(vocals):
        repeats = len(vocals) // len(song) + 1
        song = (song * repeats)[:len(vocals)]
    else:
        song = song[:len(vocals)]

    vocals = vocals.apply_gain(vocals_gain_dB)
    song   = song.apply_gain(song_gain_dB)

    combined = song.overlay(vocals, position=0)
    combined.export(output_mp3, format="mp3")
    print(f"✅ Overlay complete → {output_mp3}")

def main():
    parser = argparse.ArgumentParser(
        description="SoX + Pydub으로 Songify Meme 생성 스크립트"
    )
    parser.add_argument('voice_wav',
                        help="튜닝할 음성 WAV 파일 경로")
    parser.add_argument('song_mp3',
                        help="배경으로 사용할 노래 MP3 파일 경로")
    parser.add_argument('--cents', type=float, default=100.0,
                        help="올릴(+) 또는 내릴(-) 피치 센트 값 (기본: 100)")
    parser.add_argument('--vocals-gain', type=float, default=0.0,
                        help="튜닝된 음성 볼륨 조정 (dB, 기본 0)")
    parser.add_argument('--song-gain', type=float, default=-6.0,
                        help="배경 노래 볼륨 조정 (dB, 기본 -6)")
    parser.add_argument('--shifted-wav', default='shifted.wav',
                        help="중간 결과 피치 쉬프트 WAV 파일명")
    parser.add_argument('--output', default='songified.mp3',
                        help="최종 출력 MP3 파일명")
    args = parser.parse_args()

    # 1) SoX로 피치 쉬프트
    apply_pitch_sox(
        input_wav=args.voice_wav,
        output_wav=args.shifted_wav,
        cents=args.cents
    )

    # 2) Pydub으로 오버레이 & MP3 출력
    overlay_with_song(
        vocals_wav=args.shifted_wav,
        song_mp3=args.song_mp3,
        output_mp3=args.output,
        vocals_gain_dB=args.vocals_gain,
        song_gain_dB=args.song_gain
    )

if __name__ == '__main__':
    main()
