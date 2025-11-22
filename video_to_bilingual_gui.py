import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from faster_whisper import WhisperModel
from datetime import timedelta
from googletrans import Translator

MODEL_SIZE = "small"
SAMPLE_RATE = 16000
LANG_SOURCE = "de"
LANG_TARGET = "zh-cn"

def srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def extract_audio(video_path):
    audio_path = video_path + ".wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

def process_video(video_path, status_label):
    status_label.config(text="正在加载模型，请稍候...")
    window.update()

    model = WhisperModel(MODEL_SIZE, device="cpu")
    translator = Translator()

    status_label.config(text="正在提取音频...")
    window.update()
    audio_path = extract_audio(video_path)

    status_label.config(text="正在识别音频中...")
    window.update()
    segments, _ = model.transcribe(audio_path, beam_size=5, language=LANG_SOURCE)

    base_name = os.path.splitext(video_path)[0]
    srt_path = base_name + ".srt"

    status_label.config(text="正在翻译并生成字幕...")
    window.update()

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = srt_timestamp(seg.start)
            end = srt_timestamp(seg.end)
            orig = seg.text.strip()
            try:
                zh = translator.translate(orig, src=LANG_SOURCE, dest=LANG_TARGET).text
            except:
                zh = ""

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(orig + "\n")
            f.write(zh + "\n\n")

    status_label.config(text="处理完成！")
    messagebox.showinfo("完成", f"字幕已生成：\n{srt_path}")

def choose_video():
    video_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi")]
    )
    if video_path:
        status_label.config(text="开始处理...")
        process_video(video_path, status_label)

# GUI ------------------------

window = tk.Tk()
window.title("德语 → 中文 双语字幕生成器")
window.geometry("420x200")

label = tk.Label(window, text="请选择要处理的视频文件：", font=("Arial", 12))
label.pack(pady=10)

btn = tk.Button(window, text="选择视频", font=("Arial", 14), command=choose_video)
btn.pack(pady=10)

status_label = tk.Label(window, text="", font=("Arial", 11))
status_label.pack(pady=10)

window.mainloop()
