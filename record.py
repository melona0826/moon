import sounddevice as sd
from scipy.io.wavfile import write

samplerate = 44100
duration = 30
filename = "output2.wav"

print("녹음 시작...")
recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
write(filename, samplerate, recording)
print(f"{filename} 저장 완료.")
