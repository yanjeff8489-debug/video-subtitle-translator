#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper + 翻译引擎（DeepL / gpt-4o-mini）+ GUI API Key 设置 + 底部进度条
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
from typing import List, Optional

# speech recognition
from faster_whisper import WhisperModel

# OpenAI client (for gpt-4o-mini)
from openai import OpenAI

# DeepL SDK
try:
    import deepl
except Exception:
    deepl = None  # will handle at runtime

# -------------------------
# app base dir & config
# -------------------------
def app_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(app_base_dir(), "config.ini")

# -------------------------
# read/write API keys (both OpenAI & DeepL)
# -------------------------
def load_config():
    cfg = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        cfg.read(CONFIG_FILE, encoding="utf-8")
        result = {}
        if cfg.has_section("openai"):
            result["openai_api_key"] = cfg.get("openai", "api_key", fallback=None)
        if cfg.has_section("deepl"):
            result["deepl_api_key"] = cfg.get("deepl", "api_key", fallback=None)
        return result
    except Exception:
        return {}

def save_openai_key(key: str):
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        cfg.read(CONFIG_FILE, encoding="utf-8")
    if not cfg.has_section("openai"):
        cfg.add_section("openai")
    cfg.set("openai", "api_key", key)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        cfg.write(f)

def save_deepl_key(key: str):
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        cfg.read(CONFIG_FILE, encoding="utf-8")
    if not cfg.has_section("deepl"):
        cfg.add_section("deepl")
    cfg.set("deepl", "api_key", key)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        cfg.write(f)

# -------------------------
# status helper (must be defined before calls)
# -------------------------
def status(msg: str):
    try:
        status_label.config(text=msg)
        window.update_idletasks()
    except Exception:
        # if UI not ready, just print
        print("STATUS:", msg)

# -------------------------
# OpenAI client (lazy init)
# -------------------------
openai_client: Optional[OpenAI] = None

def init_openai_client():
    global openai_client
    cfg = load_config()
    key = cfg.get("openai_api_key")
    if not key:
        return False, "未配置 OpenAI API Key"
    try:
        openai_client = OpenAI(api_key=key)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------------------
# DeepL client (lazy init)
# -------------------------
deepl_client = None

def init_deepl_client():
    global deepl_client
    if deepl is None:
        return False, "deepl SDK 未安装，请 pip install deepl"
    cfg = load_config()
    key = cfg.get("deepl_api_key")
    if not key:
        return False, "未配置 DeepL API Key"
    try:
        deepl_client = deepl.Translator(key)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------------------
# ffmpeg helper (PyInstaller safe)
# -------------------------
def get_ffmpeg_path():
    if hasattr(sys, "_MEIPASS"):
        candidate = os.path.join(sys._MEIPASS, "ffmpeg.exe")
        if os.path.exists(candidate):
            return candidate
    exe_dir = os.path.dirname(sys.executable) if getattr(sys,"frozen",False) else os.getcwd()
    candidate2 = os.path.join(exe_dir, "ffmpeg.exe")
    if os.path.exists(candidate2):
        return candidate2
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
# translate helpers
# -------------------------

# --- DeepL batch translate (sequential with retries, updates progress) ---
def translate_with_deepl_batch(texts: List[str], target_lang="ZH", retries=3, sleep_between=0.1) -> List[str]:
    ok, err = init_deepl_client()
    if not ok:
        return ["【DeepL 未配置】" for _ in texts]
    global deepl_client
    results = []
    total = len(texts)
    done = 0
    for t in texts:
        attempt = 0
        translated = None
        while attempt < retries:
            try:
                # DeepL Translator.translate_text returns a Text object with .text
                res = deepl_client.translate_text(t, target_lang=target_lang)
                translated = res.text
                break
            except Exception as e:
                attempt += 1
                time.sleep(0.5 * attempt)
        if translated is None:
            translated = "【翻译失败】" + t
        results.append(translated)
        done += 1
        update_translation_progress(done)
        # small pause to avoid rate limit bursts
        time.sleep(sleep_between)
    return results

# --- OpenAI gpt-4o-mini batch (existing approach) ---
def translate_batch_with_context_gpt(texts: List[str], max_batch_chars=2000) -> List[str]:
    if openai_client is None:
        return ["【未配置 OpenAI Key】" for _ in texts]
    results = []
    n = len(texts)
    i = 0
    while i < n:
        chunk_texts = []
        chunk_chars = 0
        while i < n and (chunk_chars + len(texts[i])) <= max_batch_chars and len(chunk_texts) < 20:
            chunk_texts.append(texts[i])
            chunk_chars += len(texts[i])
            i += 1
        if not chunk_texts:
            chunk_texts.append(texts[i]); i+=1
        sep = "\n<<SPLIT>>\n"
        merged = sep.join(chunk_texts)
        prompt_system = (
            "你是一名专业的视频字幕翻译员。下面是多条需要翻译的句子，请按原顺序把每一条翻译成自然、流畅、简体中文。"
            "返回严格的 JSON 数组，不要多说话，数组中每个元素对应一条翻译文本。"
        )
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": merged}
                ],
                temperature=0.12,
                max_tokens=4000
            )
            out = resp.choices[0].message.content.strip()
            parsed = None
            try:
                parsed = json.loads(out)
            except Exception:
                try:
                    jstart = out.rfind('[')
                    parsed = json.loads(out[jstart:])
                except Exception:
                    parsed = None
            if isinstance(parsed, list) and len(parsed) == len(chunk_texts):
                results.extend(parsed)
            else:
                for s in chunk_texts:
                    results.append(_translate_single_retry(s))
        except Exception:
            for s in chunk_texts:
                results.append(_translate_single_retry(s))
        update_translation_progress(len(results))
    return results

def _translate_single_retry(text, retries=3, backoff=1.2):
    if openai_client is None:
        return "【未配置 OpenAI Key】" + text
    sys_prompt = "你是一名专业的视频字幕翻译员，请把下面文本翻译为自然、流畅的简体中文。"
    for attempt in range(retries):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys_prompt}, {"role":"user","content":text}],
                temperature=0.12,
                max_tokens=1000
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(backoff ** attempt)
    return "【翻译失败】" + text

# -------------------------
# SRT write
# -------------------------
def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def write_srt_bilingual(out_path: str, segments, translations):
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
    progress_bar['value'] = min(done_count, progress_bar['maximum'])
    progress_bar.update_idletasks()

# -------------------------
# main flow (threaded)
# -------------------------
def process_video_flow(video_path):
    try:
        # ensure user chose engine and keys
        engine = engine_var.get()  # "DeepL" or "ChatGPT"
        if engine == "DeepL":
            ok, err = init_deepl_client()
            if not ok:
                messagebox.showwarning("DeepL Key 问题", f"DeepL 未配置或初始化失败：{err}")
                status("DeepL 未配置")
                return
        else:
            ok, err = init_openai_client()
            if not ok:
                messagebox.showwarning("OpenAI Key 问题", f"OpenAI 未配置或初始化失败：{err}")
                status("OpenAI 未配置")
                return

        # choose model dir
        model_dir = choose_model_dir()
        if not model_dir:
            status("未选择模型")
            return
        model = load_whisper_model(model_dir)
        if model is None:
            return

        # extract audio
        audio = extract_audio(video_path)
        if not audio:
            return

        # recognition
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
        status(f"识别完成，共 {total} 段。开始翻译（{engine}）...")
        window.update()

        # translate according to engine selected
        if engine == "DeepL":
            translations = translate_with_deepl_batch(originals, target_lang="ZH", retries=3, sleep_between=0.08)
        else:
            # ChatGPT gpt-4o-mini
            ok, err = init_openai_client()
            if not ok:
                messagebox.showwarning("OpenAI Key 问题", f"OpenAI 未配置或初始化失败：{err}")
                status("OpenAI 未配置")
                return
            translations = translate_batch_with_context_gpt(originals, max_batch_chars=1800)

        if len(translations) < total:
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

def on_set_openai_key():
    k = simpledialog.askstring("设置 OpenAI API Key", "请输入 OpenAI API Key（sk-...）:", show="*")
    if k:
        save_openai_key(k.strip())
        ok, err = init_openai_client()
        if ok:
            messagebox.showinfo("成功", "OpenAI API Key 已保存并初始化成功。")
            status("OpenAI Key 已配置")
        else:
            messagebox.showwarning("注意", f"OpenAI Key 已保存，但初始化失败：{err}")

def on_set_deepl_key():
    k = simpledialog.askstring("设置 DeepL API Key", "请输入 DeepL API Key（Auth Key）:", show="*")
    if k:
        save_deepl_key(k.strip())
        ok, err = init_deepl_client()
        if ok:
            messagebox.showinfo("成功", "DeepL API Key 已保存并初始化成功。")
            status("DeepL Key 已配置")
        else:
            messagebox.showwarning("注意", f"DeepL Key 已保存，但初始化失败：{err}")

# -------------------------
# GUI layout (with single bottom progress bar)
# -------------------------
window = tk.Tk()
window.title("字幕工具 — Whisper + DeepL/gpt-4o-mini 翻译")
window.geometry("820x520")

# top: API key buttons
top_frame = tk.Frame(window)
top_frame.pack(pady=8)
tk.Button(top_frame, text="设置 OpenAI API Key", command=on_set_openai_key, font=("Microsoft YaHei", 10)).pack(side="left", padx=6)
tk.Button(top_frame, text="设置 DeepL API Key", command=on_set_deepl_key, font=("Microsoft YaHei", 10)).pack(side="left", padx=6)
tk.Button(top_frame, text="打开 config.ini", command=lambda: os.startfile(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else messagebox.showinfo("提示","config.ini 不存在"), font=("Microsoft YaHei", 10)).pack(side="left", padx=6)

# engine selection (no auto-switch)
engine_frame = tk.Frame(window)
engine_frame.pack(pady=6)
tk.Label(engine_frame, text="选择翻译引擎（不自动切换）：", font=("Microsoft YaHei", 12)).pack(side="left")
engine_var = tk.StringVar(value="DeepL")
engine_box = ttk.Combobox(engine_frame, textvariable=engine_var, state="readonly", width=16,
                          values=["DeepL", "ChatGPT (gpt-4o-mini)"])
engine_box.pack(side="left", padx=6)

# language selection
lang_frame = tk.Frame(window)
lang_frame.pack(pady=6)
tk.Label(lang_frame, text="识别语言：", font=("Microsoft YaHei", 12)).pack(side="left")
language_var = tk.StringVar(value="auto")
language_box = ttk.Combobox(lang_frame, textvariable=language_var, state="readonly", width=12,
                           values=["auto","de","en","zh","fr","es","fi","ja"])
language_box.pack(side="left", padx=6)

# center: choose video and start
center = tk.Frame(window)
center.pack(pady=20)
tk.Button(center, text="选择视频并开始", command=choose_video_and_start, font=("Microsoft YaHei", 14), width=28).pack()

# status label
status_label = tk.Label(window, text="准备就绪", font=("Microsoft YaHei", 11), wraplength=780, justify="left")
status_label.pack(pady=8)

# bottom single progress bar
progress_bar = ttk.Progressbar(window, orient="horizontal", length=780, mode="determinate")
progress_bar.pack(pady=10)

# help text
help_text = (
    "说明：\n"
    "1) 首次使用请点击“设置 DeepL API Key”或“设置 OpenAI API Key”。\n"
    "2) 运行时会提示选择 Whisper 模型目录（如 small / medium）。\n"
    "3) 若使用 DeepL，请确保你的 DeepL Key 有权限调用 API（Auth Key）。\n"
    "4) 程序会输出 *_bilingual.srt 文件（原文 + 中文）。"
)
tk.Label(window, text=help_text, font=("Microsoft YaHei", 10), fg="#333", justify="left").pack(padx=12)

# initialize clients (if keys present)
cfg = load_config()
if cfg.get("openai_api_key"):
    try:
        init_openai_client()
    except:
        pass
if cfg.get("deepl_api_key"):
    try:
        init_deepl_client()
    except:
        pass

if not cfg:
    status("未检测到 API Key，请点击“设置 DeepL API Key”或“设置 OpenAI API Key”")

window.mainloop()
