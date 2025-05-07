import sounddevice as sd
from scipy.io.wavfile import write

samplerate = 44100  # 샘플링 주파수 (Hz)
duration = 120        # 녹음 시간 (초)
filename = "output.wav"

print("녹음 시작...")
recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
write(filename, samplerate, recording)
print(f"{filename} 저장 완료.")
