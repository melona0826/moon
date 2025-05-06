#!/usr/bin/env python3
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment

def extract_audio(video_path: str, audio_path: str):
    """비디오에서 오디오만 WAV로 추출"""
    videoclip = VideoFileClip(video_path)
    videoclip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    videoclip.close()

def apply_autotune(input_wav: str,
                   output_wav: str,
                   autotalent_so: str = '/usr/lib/ladspa/autotalent.so',
                   key: str = 'C',
                   scale: str = 'major',
                   shift: float = 0.0,
                   amount: float = 100.0,
                   mix: float = 100.0):
    """
    FFmpeg+LADSPA Autotalent으로 오토튠 적용
    - shift: 음정 이동(cent 단위), amount: 튠 강도(%), mix: dry/wet 비율(%)
    - ladspa 파라미터 참조: https://ccrma.stanford.edu/software/ladspa/plugins/lib/autotalent.html
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

def overlay_beat(vocals_wav: str,
                 beat_wav: str,
                 output_wav: str,
                 beat_gain_dB: float = -6.0):
    """드럼 루프(beat)를 보컬 위에 루프 오버레이"""
    vocals = AudioSegment.from_file(vocals_wav)
    beat    = AudioSegment.from_file(beat_wav)
    # 보컬 길이에 맞춰 비트 반복
    repeats = len(vocals) // len(beat) + 1
    beat_loop = (beat * repeats)[:len(vocals)].apply_gain(beat_gain_dB)
    combined = vocals.overlay(beat_loop)
    combined.export(output_wav, format='wav')

def combine_video_audio(original_video: str,
                        final_audio: str,
                        output_video: str):
    """원본 비디오에 최종 오디오 합치기"""
    video = VideoFileClip(original_video)
    audio = AudioFileClip(final_audio)
    final = video.set_audio(audio)
    final.write_videofile(output_video,
                          codec='libx264',
                          audio_codec='aac',
                          temp_audiofile='temp-audio.m4a',
                          remove_temp=True)
    video.close()
    audio.close()

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description="간단한 Songify Meme 생성 스크립트")
    p.add_argument('input_video', help="원본 영상 파일 경로")
    p.add_argument('beat_wav',     help="드럼 루프 WAV 파일 경로")
    p.add_argument('--key',        default='C',      help="튜닝 키 (예: C, D#, F)")
    p.add_argument('--scale',      default='major',  help="스케일 (major/minor)")
    p.add_argument('--shift',      type=float, default=0.0,
                   help="전체 피치 이동(cent 단위)")
    p.add_argument('--amount',     type=float, default=100.0,
                   help="튜닝 강도 (%)")
    p.add_argument('--mix',        type=float, default=100.0,
                   help="Dry/Wet 믹스 (%)")
    p.add_argument('--autotalent', default='/usr/lib/ladspa/autotalent.so',
                   help="Autotalent LADSPA 플러그인 경로")
    p.add_argument('--output',     default='songified.mp4',
                   help="출력 비디오 파일명")
    args = p.parse_args()

    # 1) 오디오 추출
    extract_audio(args.input_video, 'original.wav')

    # 2) 오토튠
    apply_autotune('original.wav',
                   'tuned.wav',
                   autotalent_so=args.autotalent,
                   key=args.key,
                   scale=args.scale,
                   shift=args.shift,
                   amount=args.amount,
                   mix=args.mix)

    # 3) 비트 오버레이
    overlay_beat('tuned.wav', args.beat_wav, 'final_audio.wav')

    # 4) 영상 + 오디오 합치기
    combine_video_audio(args.input_video, 'final_audio.wav', args.output)

    print(f"✅ 완료! 생성된 파일: {args.output}")
