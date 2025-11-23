import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from faster_whisper import WhisperModel
from googletrans import Translator


# ---------------------------------------------------
# ffmpeg 路径处理（支持 PyInstaller onefile）
# ---------------------------------------------------
def get_ffmpeg_path():
    if hasattr(sys, "_MEIPASS"):  # PyInstaller 解压目录
        return os.path.join(sys._MEIPASS, "ffmpeg.exe")
    return "ffmpeg.exe"  # 普通运行模式


# ---------------------------------------------------
# 全局翻译器
# ---------------------------------------------------
translator = Translator()


# ---------------------------------------------------
# 加载 Whisper 模型（从用户选择的目录）
# ---------------------------------------------------
def load_model():
    status_label.config(text="请选择 Whisper 模型目录（如 small 或 medium）…")
    window.update()

    model_dir = filedialog.askdirectory(title="请选择 Whisper 模型文件夹")
    if not model_dir:
        status_label.config(text="未选择模型，操作取消")
        return None

    status_label.config(text="正在加载模型，请稍候…（large/medium 模型可能较慢）")
    window.update()

    try:
        model = WhisperModel(
            model_dir,
            device="cpu",
            compute_type="float32"  # 保持原始精度（不 int8）
        )
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
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    except Exception as e:
        messagebox.showerror("错误", f"ffmpeg 执行失败：\n{e}")
        return None

    return audio_path


# ---------------------------------------------------
# 生成双语字幕文件
# ---------------------------------------------------
def write_srt(output_path, segments):
    status_label.config(text="正在生成双语字幕…")
    window.update()

    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            start = seg.start
            end = seg.end
            text = seg.text.strip()

            start_time = timestamp(start)
            end_time = timestamp(end)

            try:
                zh = translator.translate(text, dest="zh-cn").text
            except:
                zh = "【翻译失败】 " + text

            f.write(f"{idx}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n{zh}\n\n")
            idx += 1


def timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# ---------------------------------------------------
# 主流程：选择视频 → 加载模型 → 识别 → 生成字幕
# ---------------------------------------------------
def process_video(video_path):
    model = load_model()
    if model is None:
        return

    audio_path = extract_audio(video_path)
    if audio_path is None:
        return

    # 语言设置
    lang_value = language_var.get()
    if lang_value == "auto":
        lang = None
        status_label.config(text="正在识别语音（自动检测语言）…")
    else:
        lang = lang_value
        status_label.config(text=f"正在识别语音（指定语言：{lang}）…")

    window.update()

    # Whisper 转写
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=lang
    )
    segments = list(segments)

    # 输出字幕
    out_srt = video_path + "_bilingual.srt"
    write_srt(out_srt, segments)

    status_label.config(text=f"字幕生成成功：\n{out_srt}")
    messagebox.showinfo("完成", f"已生成字幕：\n{out_srt}")


# ---------------------------------------------------
# 多线程避免 GUI 卡死
# ---------------------------------------------------
def process_in_thread(video_path):
    thread = threading.Thread(target=process_video, args=(video_path,), daemon=True)
    thread.start()


# ---------------------------------------------------
# GUI 控件：选择视频
# ---------------------------------------------------
def choose_video():
    video_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4;*.mkv;*.avi;*.mov")]
    )
    if video_path:
        status_label.config(text="开始处理视频…")
        process_in_thread(video_path)


# ---------------------------------------------------
# GUI 主界面
# ---------------------------------------------------
window = tk.Tk()
window.title("Whisper 视频语音识别 + 双语字幕生成工具")
window.geometry("650x420")

tk.Label(window, text="选择识别语言：", font=("Microsoft YaHei", 12)).pack()

# 简洁语言列表 + 芬兰语 + 日语
language_var = tk.StringVar()
language_combo = ttk.Combobox(
    window,
    textvariable=language_var,
    values=["auto", "de", "en", "zh", "fr", "es", "fi", "ja"],
    state="readonly",
    width=20,
    font=("Microsoft YaHei", 12)
)
language_combo.pack(pady=5)
language_combo.current(0)

tk.Button(
    window,
    text="选择视频文件",
    font=("Microsoft YaHei", 16),
    command=choose_video
).pack(pady=20)

status_label = tk.Label(
    window, text="准备就绪", font=("Microsoft YaHei", 12), wraplength=600
)
status_label.pack(pady=20)

window.mainloop()
