import deepspeech
import wave
import numpy as np

# โหลดโมเดล
model_path = "deepspeech-0.9.3-models.pbmm"
scorer_path = "deepspeech-0.9.3-models.scorer"

model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)

# โหลดไฟล์เสียง (ต้องเป็น WAV 16-bit PCM)
audio_file = "audio.wav"
with wave.open(audio_file, "rb") as wf:
    frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)

# แปลงเสียงเป็นข้อความ
text = model.stt(audio)
print("ผลลัพธ์: ", text)
