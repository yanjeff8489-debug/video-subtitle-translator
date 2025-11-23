import os
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from faster_whisper import WhisperModel
from googletrans import Translator


# -------------------------
# Global Translator
# -------------------------
translator = Translator()


# -------------------------
# Load Whisper model (user chooses model directory)
# -------------------------
def load_model():
    status_label.config(text="请选择 Whisper 模型文件夹（small 或 medium）…")
    window.update()

    model_dir = filedialog.askdirectory(title="请选择模型文件夹")
    if not model_dir:
        status_label.config(text="未选择模型，操作取消")
        return None

    status_label.config(text="正在加载模型，请稍候…（可能需要 10~30 秒）")
    window.update()

    try:
        model = WhisperModel(
            model_dir,
            device="cpu",
            compute_type="float32"  # 保持原始精度
        )
        status_label.config(text="模型加载成功！")
        return model

    except Exception as e:
        messagebox.showerror("错误", f"模型加载失败：\n{e}")
        status_label.config(text="模型加载失败")
        return None


# -------------------------
# FFMPEG extract audio
# -------------------------
def extract_audio(video_path):
    status_label.config(text="正在提取音频…")
    window.update()

    audio_path = video_path + "_audio.wav"
    ffmpeg_path = "ffmpeg.exe"

    cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-y", audio_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    except Exception as e:
        messagebox.showerror("错误", f"执行 ffmpeg 时出错：\n{e}")
        return None

    return audio_path


# -------------------------
# Write bilingual subtitles
# -------------------------
def write_srt(output_path, segments):
    status_label.config(text="正在生成双语字幕…")
    window.update()

    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            start = seg.start
            end = seg.end
            text = seg.text.strip()

            start_time = format_timestamp(start)
            end_time = format_timestamp(end)

            try:
                zh = translator.translate(text, dest="zh-cn").text
            except:
                zh = "【翻译失败】" + text

            f.write(f"{idx}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n{zh}\n\n")
            idx += 1


def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# -------------------------
# Main processing
# -------------------------
def process_video(video_path):
    # Load model
    model = load_model()
    if model is None:
        return

    # Extract audio
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return

    # Get selected language
    lang_mode = language_var.get()
    if lang_mode == "auto":
        language = None
        status_label.config(text="正在识别语音（自动检测语言）…")
    else:
        language = lang_mode
        status_label.config(text=f"正在识别语音（指定语言：{lang_mode}）…")

    window.update()

    # Whisper transcription
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language
    )
    segments = list(segments)

    # Write subtitles
    output_srt = video_path + "_bilingual.srt"
    write_srt(output_srt, segments)

    status_label.config(text=f"字幕生成完成：\n{output_srt}")
    messagebox.showinfo("完成", f"已生成双语字幕：\n{output_srt}")


# -------------------------
# Run in thread
# -------------------------
def process_video_thread(video_path):
    thread = threading.Thread(target=process_video, args=(video_path,), daemon=True)
    thread.start()


# -------------------------
# Button callback
# -------------------------
def choose_video():
    video_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("Video Files", "*.mp4;*.mkv;*.avi;*.mov")]
    )
    if video_path:
        status_label.config(text="开始处理视频…")
        process_video_thread(video_path)


# -------------------------
# GUI
# -------------------------
window = tk.Tk()
window.title("视频转文字并生成双语字幕（Whisper）")
window.geometry("620x380")

# 语言选择框（包括芬兰语、日语）
tk.Label(window, text="选择识别语言：", font=("Microsoft YaHei", 12)).pack()

language_var = tk.StringVar()
language_dropdown = ttk.Combobox(
    window,
    textvariable=language_var,
    values=[
        ("auto"),   # 自动检测
        ("de"),     # 德语
        ("en"),     # 英语
        ("zh"),     # 中文
        ("fr"),     # 法语
        ("es"),     # 西班牙语
        ("fi"),     # 芬兰语
        ("ja")      # 日语
    ],
    state="readonly",
    font=("Microsoft YaHei", 12),
    width=20
)
language_dropdown.pack(pady=5)
language_dropdown.current(0)   # 默认自动检测

# 视频选择按钮
choose_btn = tk.Button(window, text="选择视频文件", font=("Microsoft YaHei", 16), command=choose_video)
choose_btn.pack(pady=20)

status_label = tk.Label(window, text="准备就绪", font=("Microsoft YaHei", 12), wraplength=580)
status_label.pack(pady=20)

window.mainloop()
