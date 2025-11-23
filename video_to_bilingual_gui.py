import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from faster_whisper import WhisperModel
import deepl


# ---------------------------------------------------
# ffmpeg 路径处理（支持 PyInstaller onefile）
# ---------------------------------------------------
def get_ffmpeg_path():
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "ffmpeg.exe")
    return "ffmpeg.exe"


# ---------------------------------------------------
# 初始化 DeepL 翻译器
# ---------------------------------------------------
DEEPL_API_KEY = "在这里填入你的 DeepL API Key"

try:
    translator = deepl.Translator(DEEPL_API_KEY)
except Exception as e:
    translator = None
    print("DeepL 初始化失败：", e)


# ---------------------------------------------------
# 使用 DeepL 翻译
# ---------------------------------------------------
def translate_text(text):
    if translator is None:
        return "【翻译失败】" + text

    try:
        result = translator.translate_text(text, target_lang="ZH")
        return result.text
    except Exception:
        return "【翻译失败】" + text


# ---------------------------------------------------
# 选择模型目录并加载模型
# ---------------------------------------------------
def load_model():
    status_label.config(text="请选择 Whisper 模型目录（如 small 或 medium）…")
    window.update()

    model_dir = filedialog.askdirectory(title="选择 Whisper 模型文件夹")
    if not model_dir:
        status_label.config(text="未选择模型，操作取消")
        return None

    status_label.config(text="正在加载模型，请稍候…")
    window.update()

    try:
        model = WhisperModel(model_dir, device="cpu", compute_type="float32")
        status_label.config(text="模型加载成功！")
        return model
    except Exception as e:
        messagebox.showerror("错误", f"模型加载失败：\n{e}")
        status_label.config(text="模型加载失败")
        return None


# ---------------------------------------------------
# 使用 ffmpeg 提取音频
# ---------------------------------------------------
def extract_audio(video_path):
    status_label.config(text="正在提取音频…")
    window.update()

    audio_path = video_path + "_audio.wav"
    ffmpeg_path = get_ffmpeg_path()

    cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-y", audio_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        messagebox.showerror("错误", f"ffmpeg 执行失败：\n{e}")
        return None

    return audio_path


# ---------------------------------------------------
# 生成双语 SRT 字幕
# ---------------------------------------------------
def write_srt(output_path, segments):
    status_label.config(text="正在生成双语字幕（DeepL 高质量翻译）…")
    window.update()

    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            start = timestamp(seg.start)
            end = timestamp(seg.end)
            original = seg.text.strip()
            chinese = translate_text(original)

            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{original}\n{chinese}\n\n")
            idx += 1


def timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# ---------------------------------------------------
# 主流程
# ---------------------------------------------------
def process_video(video_path):
    model = load_model()
    if model is None:
        return

    audio_path = extract_audio(video_path)
    if audio_path is None:
        return

    lang_value = language_var.get()
    lang = None if lang_value == "auto" else lang_value

    status_label.config(text="正在识别语音…")
    window.update()

    segments, info = model.transcribe(audio_path, beam_size=5, language=lang)
    segments = list(segments)

    out_srt = video_path + "_bilingual.srt"
    write_srt(out_srt, segments)

    status_label.config(text=f"生成成功：{out_srt}")
    messagebox.showinfo("完成", f"已生成字幕：\n{out_srt}")


# ---------------------------------------------------
# 启动线程
# ---------------------------------------------------
def process_in_thread(video_path):
    threading.Thread(target=process_video, args=(video_path,), daemon=True).start()


# ---------------------------------------------------
# GUI
# ---------------------------------------------------
def choose_video():
    video_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4;*.mkv;*.avi;*.mov")]
    )
    if video_path:
        status_label.config(text="开始处理视频…")
        process_in_thread(video_path)


window = tk.Tk()
window.title("Whisper 高质量翻译字幕生成器（DeepL 版）")
window.geometry("650x420")

tk.Label(window, text="选择识别语言：", font=("Microsoft YaHei", 12)).pack()

language_var = tk.StringVar()
language_combo = ttk.Combobox(
    window,
    textvariable=language_var,
    values=["auto", "de", "en", "zh", "fr", "es", "fi", "ja"],
    state="readonly",
    width=20,
)
language_combo.pack()
language_combo.current(0)

tk.Button(window, text="选择视频文件", font=("Microsoft YaHei", 16), command=choose_video).pack(pady=20)

status_label = tk.Label(window, text="准备就绪", font=("Microsoft YaHei", 12), wraplength=600)
status_label.pack()

window.mainloop()
