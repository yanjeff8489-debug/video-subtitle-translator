#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper + gpt-4o-mini 翻译（上下文批量）+ GUI API Key 设置 + 底部单一进度条
保存为 video_to_bilingual_gui.py
"""

import os
import sys
import json
import time
import threading
import subprocess
import configparser
import tkinter as tk
import tkinter.simpledialog as simpledialog
from tkinter import filedialog, ttk, messagebox
from typing import List

from faster_whisper import WhisperModel
from openai import OpenAI

# -------------------------
# app base dir & config
# -------------------------
def app_base_dir():
    if getattr(sys, "frozen", False):
        # exe 模式：config 放在 exe 同目录
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(app_base_dir(), "config.ini")

# -------------------------
# read/write API key
# -------------------------
def load_api_key():
    cfg = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        cfg.read(CONFIG_FILE, encoding="utf-8")
        return cfg.get("openai", "api_key", fallback=None)
    except Exception:
        return None

def save_api_key(key: str):
    cfg = configparser.ConfigParser()
    cfg["openai"] = {"api_key": key}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        cfg.write(f)

def ask_and_save_api_key():
    key = simpledialog.askstring("设置 OpenAI API Key", "请输入你的 OpenAI API Key（sk-...）:", show="*")
    if key:
        save_api_key(key.strip())
        messagebox.showinfo("已保存", "API Key 已保存到 config.ini（程序目录）。")
        return key.strip()
    return None

# -------------------------
# OpenAI init (lazy)
# -------------------------
openai_client = None

def init_openai_client():
    global openai_client
    key = load_api_key()
    if not key:
        return False, "未配置 API Key"
    try:
        openai_client = OpenAI(api_key=key)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------------------
# ffmpeg helper (PyInstaller safe)
# -------------------------
def get_ffmpeg_path():
    # first try _MEIPASS (onefile)
    if hasattr(sys, "_MEIPASS"):
        candidate = os.path.join(sys._MEIPASS, "ffmpeg.exe")
        if os.path.exists(candidate):
            return candidate
    # try exe dir
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    candidate2 = os.path.join(exe_dir, "ffmpeg.exe")
    if os.path.exists(candidate2):
        return candidate2
    # fallback to PATH name
    return "ffmpeg.exe"

# -------------------------
# model selection & load
# -------------------------
def choose_model_dir():
    d = filedialog.askdirectory(title="请选择 Whisper 模型目录（例如 small 或 medium）")
    return d if d else None

def load_whisper_model(model_dir):
    status("正在加载 Whisper 模型（可能需要较长时间）...")
    window.update()
    try:
        model = WhisperModel(model_dir, device="cpu", compute_type="float32")
        status("模型加载完成")
        return model
    except Exception as e:
        messagebox.showerror("模型加载失败", str(e))
        status("模型加载失败")
        return None

# -------------------------
# audio extract
# -------------------------
def extract_audio(video_path):
    status("正在提取音频（ffmpeg）...")
    window.update()
    audio_path = video_path + "_audio.wav"
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-y", audio_path]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        messagebox.showerror("ffmpeg 错误", f"ffmpeg 执行失败：\n{e}\n请确保 ffmpeg.exe 在 EXE 同目录或 PATH 中")
        status("ffmpeg 错误")
        return None
    except Exception as e:
        messagebox.showerror("ffmpeg 异常", str(e))
        status("ffmpeg 错误")
        return None

# -------------------------
# translate helpers (gpt-4o-mini)
# -------------------------

def translate_batch_with_context(texts: List[str], max_batch_chars=2000) -> List[str]:
    """
    更智能的上下文批量翻译策略：
    - 合并相邻短段成上下文块，但每个块字符数不超过 max_batch_chars
    - 对每个块发一次请求，要求返回 JSON 列表（对应每个子句）
    - 解析失败则回退到逐句翻译
    """
    if openai_client is None:
        return ["【未配置 API Key】" + t for t in texts]

    results = []
    n = len(texts)
    i = 0
    while i < n:
        # build a chunk
        chunk_texts = []
        chunk_chars = 0
        while i < n and (chunk_chars + len(texts[i])) <= max_batch_chars and len(chunk_texts) < 20:
            chunk_texts.append(texts[i])
            chunk_chars += len(texts[i])
            i += 1
        # if nothing collected (one very large sentence), force one
        if not chunk_texts:
            chunk_texts.append(texts[i])
            i += 1

        # merge chunk with separator and ask model to return JSON array
        sep = "\n<<SPLIT>>\n"
        merged = sep.join(chunk_texts)
        prompt_system = (
            "你是一名专业的视频字幕翻译员。下面是多条需要翻译的句子，请按原顺序把每一条翻译成自然、流畅、简体中文。"
            "  返回严格的 JSON 数组，不要多说话，数组中每个元素对应一条翻译文本。"
        )
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": merged}
                ],
                temperature=0.15,
                max_tokens=4000
            )
            out = resp.choices[0].message.content.strip()
            # try parse JSON
            parsed = None
            try:
                parsed = json.loads(out)
            except Exception:
                # try to extract last JSON in text (some models prepend explanation)
                try:
                    jstart = out.rfind('[')
                    parsed = json.loads(out[jstart:])
                except Exception:
                    parsed = None
            if isinstance(parsed, list) and len(parsed) == len(chunk_texts):
                results.extend(parsed)
            else:
                # fallback: per-sentence
                for s in chunk_texts:
                    results.append(_translate_single_retry(s))
        except Exception:
            # batch request failed, fallback
            for s in chunk_texts:
                results.append(_translate_single_retry(s))
        # update progress in GUI per-chunk
        update_translation_progress(len(results))
    return results

def _translate_single_retry(text, retries=3, backoff=1.2):
    if openai_client is None:
        return "【未配置 API Key】" + text
    sys_prompt = "你是一名专业的视频字幕翻译员，请把下面文本翻译为自然、流畅的简体中文。"
    for attempt in range(retries):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys_prompt}, {"role":"user","content":text}],
                temperature=0.15,
                max_tokens=1000
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(backoff ** attempt)
            continue
    return "【翻译失败】" + text

# -------------------------
# SRT write
# -------------------------
def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def write_srt_bilingual(out_path, segments, translations):
    status("正在写入 SRT 文件...")
    window.update()
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (seg, zh) in enumerate(zip(segments, translations), start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            orig = seg.text.strip()
            f.write(f"{idx}\n{start} --> {end}\n{orig}\n{zh}\n\n")
    status(f"SRT 已生成：{out_path}")

# -------------------------
# progress helpers (single bottom progress bar)
# -------------------------
def init_progress():
    progress_bar['mode'] = 'determinate'
    progress_bar['value'] = 0
    progress_bar.update_idletasks()

def set_progress_max(maxv):
    progress_bar['maximum'] = maxv
    progress_bar.update_idletasks()

def update_progress_step(step=1):
    progress_bar.step(step)
    progress_bar.update_idletasks()

def update_translation_progress(done_count):
    # called by batch translate to update progress
    progress_bar['value'] = min(done_count, progress_bar['maximum'])
    progress_bar.update_idletasks()

# -------------------------
# main flow (threaded)
# -------------------------
def process_video_flow(video_path):
    try:
        ok, err = init_openai_client()
        if not ok:
            messagebox.showwarning("API Key 问题", "请先设置 OpenAI API Key")
            status("未配置 API Key")
            return

        model_dir = choose_model_dir()
        if not model_dir:
            status("未选择模型")
            return
        model = load_whisper_model(model_dir)
        if model is None:
            return

        audio = extract_audio(video_path)
        if not audio:
            return

        # recognition stage (indeterminate)
        status("正在识别语音（Whisper）…")
        progress_bar['mode'] = 'indeterminate'
        progress_bar.start(10)
        window.update()

        lang_val = language_var.get()
        lang = None if lang_val == "auto" else lang_val

        segments, info = model.transcribe(audio, beam_size=5, language=lang)
        segments = list(segments)

        # stop indeterminate
        progress_bar.stop()
        progress_bar['mode'] = 'determinate'
        progress_bar['value'] = 0
        window.update()

        if not segments:
            status("识别结果为空")
            return

        originals = [s.text.strip() for s in segments]
        total = len(originals)
        set_progress_max(total)
        status(f"识别完成，共 {total} 段。开始批量上下文翻译（gpt-4o-mini）...")
        window.update()

        translations = translate_batch_with_context(originals, max_batch_chars=1800)

        # ensure translations length matches
        if len(translations) < total:
            # pad failures
            translations += ["【翻译失败】"] * (total - len(translations))

        out_srt = os.path.splitext(video_path)[0] + "_bilingual.srt"
        write_srt_bilingual(out_srt, segments, translations)

        progress_bar['value'] = progress_bar['maximum']
        status("字幕生成完成")
        messagebox.showinfo("完成", f"已生成字幕：\n{out_srt}")

    except Exception as e:
        messagebox.showerror("处理失败", str(e))
        status("处理失败")

def start_thread_for_video(video_path):
    t = threading.Thread(target=process_video_flow, args=(video_path,), daemon=True)
    t.start()

# -------------------------
# GUI callbacks
# -------------------------
def choose_video_and_start():
    path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件","*.mp4;*.mkv;*.avi;*.mov")])
    if path:
        status("准备开始...")
        start_thread_for_video(path)

def on_set_api_key():
    k = simpledialog.askstring("设置 OpenAI API Key", "请输入 OpenAI API Key（sk-...）:", show="*")
    if k:
        save_api_key(k.strip())
        ok, err = init_openai_client()
        if ok:
            messagebox.showinfo("成功", "API Key 已保存并初始化成功。")
            status("API Key 已配置")
        else:
            messagebox.showwarning("注意", f"API Key 已保存，但初始化失败：{err}")

# -------------------------
# GUI layout (with single bottom progress bar)
# -------------------------
window = tk.Tk()
window.title("字幕工具 — Whisper + gpt-4o-mini 翻译（进度条底部）")
window.geometry("760x460")

top_frame = tk.Frame(window)
top_frame.pack(pady=8)
tk.Button(top_frame, text="设置 API Key", command=on_set_api_key, font=("Microsoft YaHei", 10)).pack(side="left", padx=6)
tk.Button(top_frame, text="打开 config.ini", command=lambda: os.startfile(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else messagebox.showinfo("提示","config.ini 不存在"), font=("Microsoft YaHei", 10)).pack(side="left", padx=6)

lang_frame = tk.Frame(window)
lang_frame.pack(pady=6)
tk.Label(lang_frame, text="识别语言：", font=("Microsoft YaHei", 12)).pack(side="left")
language_var = tk.StringVar(value="auto")
language_box = ttk.Combobox(lang_frame, textvariable=language_var, state="readonly", width=12,
                           values=["auto","de","en","zh","fr","es","fi","ja"])
language_box.pack(side="left", padx=6)

center = tk.Frame(window)
center.pack(pady=20)
tk.Button(center, text="选择视频并开始", command=choose_video_and_start, font=("Microsoft YaHei", 14), width=24).pack()

status_label = tk.Label(window, text="准备就绪", font=("Microsoft YaHei", 11), wraplength=720, justify="left")
status_label.pack(pady=8)

# bottom single progress bar (Option 1)
progress_bar = ttk.Progressbar(window, orient="horizontal", length=700, mode="determinate")
progress_bar.pack(pady=10)

help_text = (
    "说明：\n"
    "1) 首次使用请点击“设置 API Key”并输入你的 OpenAI API Key。\n"
    "2) 运行时会提示选择 Whisper 模型目录（如 small / medium）。\n"
    "3) 请确保 ffmpeg.exe 与 EXE 同目录，或已在 PATH 中。\n"
    "4) 程序会输出 *_bilingual.srt 文件（原文 + 中文）。"
)
tk.Label(window, text=help_text, font=("Microsoft YaHei", 10), fg="#333", justify="left").pack(padx=12)

# initialize
_init_ok, _err = init_openai_client()
if not _init_ok:
    status("未检测到 API Key，请点击“设置 API Key”")
else:
    status("已检测到 API Key（可直接开始）")

window.mainloop()
