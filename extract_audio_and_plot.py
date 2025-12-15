# extract_audio_and_plot.py
# -*- coding: utf-8 -*-
"""
从 .mov 视频中剥离音频为 .wav 并进行时域波形可视化（整段 + 放大预览）
依赖: ffmpeg 可执行文件、Python: numpy, soundfile, librosa, matplotlib, tqdm

安装依赖:
  pip install numpy soundfile librosa matplotlib tqdm

示例:
  python extract_audio_and_plot.py --video input.mov
  # 若在 Windows 且 ffmpeg 不在 PATH：
  python extract_audio_and_plot.py --video input.mov --ffmpeg "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
  # 指定输出目录与采样率：
  
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_ffmpeg_extract_wav(ffmpeg_path, in_mov, out_wav, sr=48000, overwrite=True):
    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",
        "-i", str(in_mov),
        "-vn",                # 不要视频
        "-ac", "1",           # 单声道（便于画波形）
        "-ar", str(sr),       # 目标采样率
        "-f", "wav",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "未找到 ffmpeg 可执行文件。请安装 ffmpeg 或通过 --ffmpeg 指定路径。"
        )
    except subprocess.CalledProcessError as e:
        # 打印 ffmpeg 错误信息便于排查
        sys.stderr.write(e.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg 提取音频失败，请检查输入文件与参数。")


def plot_waveform(y, sr, out_png_full, out_png_zoom=None, zoom_sec=2.0):
    """
    画两张图：
      1) 全时长波形（整段）
      2) 开头 zoom_sec 秒的放大波形（可选）
    """
    duration = len(y) / sr
    t = np.arange(len(y)) / sr

    # 全段
    plt.figure(figsize=(14, 3.6))
    plt.title(f"Waveform (Full) - duration = {duration:.2f}s, sr = {sr} Hz")
    plt.plot(t, y, linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_png_full, dpi=150)
    plt.close()

    # 放大预览
    if out_png_zoom is not None and duration > 0:
        zoom_N = int(min(len(y), zoom_sec * sr))
        t_zoom = t[:zoom_N]
        y_zoom = y[:zoom_N]
        plt.figure(figsize=(14, 3.6))
        plt.title(f"Waveform (Zoomed First {zoom_sec:.2f}s)")
        plt.plot(t_zoom, y_zoom, linewidth=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(out_png_zoom, dpi=150)
        plt.close()


def save_wave_csv(y, sr, out_csv, step=100):
    """
    采样下采样后保存 CSV（减少体积）：每 step 个点取一个
    CSV 列含义：time_sec, amplitude
    """
    idx = np.arange(0, len(y), step, dtype=int)
    t = idx / sr
    data = np.column_stack([t, y[idx]])
    header = "time_sec,amplitude"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.6f")


def main():
    parser = argparse.ArgumentParser(description="从 .mov 提取音频并进行波形可视化")
    parser.add_argument("--video", required=True, help="输入 .mov 视频路径")
    parser.add_argument("--out", default="audio_waveform_output", help="输出目录")
    parser.add_argument("--sr", type=int, default=48000, help="目标采样率（Hz），默认 48000")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件路径或已在 PATH 的命令名")
    parser.add_argument("--no_csv", action="store_true", help="不导出下采样 CSV")
    parser.add_argument("--zoom_sec", type=float, default=2.0, help="放大预览时长（秒）")
    args = parser.parse_args()

    in_mov = Path(args.video).resolve()
    if not in_mov.exists():
        print(f"输入文件不存在: {in_mov}")
        sys.exit(1)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_wav = out_dir / (in_mov.stem + ".wav")
    out_png_full = out_dir / (in_mov.stem + "_wave_full.png")
    out_png_zoom = out_dir / (in_mov.stem + f"_wave_zoom_{int(args.zoom_sec)}s.png")
    out_csv = out_dir / (in_mov.stem + "_wave_downsampled.csv")

    print(">>> 使用 ffmpeg 提取音频为 WAV ...")
    run_ffmpeg_extract_wav(args.ffmpeg, in_mov, out_wav, sr=args.sr, overwrite=True)
    print(f"音频已导出: {out_wav}")

    print(">>> 读取 WAV 并生成波形图 ...")
    # 直接用 soundfile 读取（速度快、节省内存），必要时转 float32
    y, sr = sf.read(str(out_wav), dtype="float32", always_2d=False)
    # 若读出是多通道，转单声道（平均）
    if y.ndim > 1:
        y = y.mean(axis=1)

    # 可选：轻微去 DC 偏置
    y = y - np.mean(y)

    # 可选：归一化到 [-1, 1]（避免极端幅值）
    max_abs = np.max(np.abs(y)) if len(y) else 1.0
    if max_abs > 0:
        y = y / max_abs

    plot_waveform(y, sr, out_png_full, out_png_zoom, zoom_sec=args.zoom_sec)
    print(f"波形图(整段)已保存: {out_png_full}")
    print(f"波形图(放大{args.zoom_sec:.0f}s)已保存: {out_png_zoom}")

    if not args.no_csv:
        print(">>> 导出下采样 CSV（便于快速查看/绘图） ...")
        save_wave_csv(y, sr, out_csv, step=100)
        print(f"CSV 已保存: {out_csv}")

    print("\n完成 ✅")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
