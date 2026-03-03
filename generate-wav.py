import wave
import math
import struct

sample_rate = 24000
duration = 2.0
frequency = 440.0
amplitude = 0.1  # 0.0 to 1.0

n_samples = int(sample_rate * duration)

with wave.open("test.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(sample_rate)

    for i in range(n_samples):
        t = i / sample_rate
        sample = amplitude * math.sin(2 * math.pi * frequency * t)
        wf.writeframes(struct.pack("<h", int(sample * 32767)))

print("wrote test.wav")
