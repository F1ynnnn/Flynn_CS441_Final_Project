# -*- coding: utf-8 -*-
"""
音频驱动的击球事件检测 + 导出每次击球的音频片段 + 对应视频窗口内“球最大”的10帧

依赖:
  pip install librosa numpy scipy opencv-python pandas tqdm
  (可选) ffmpeg 可执行文件路径 Windows 默认 C:\ProgramData\chocolatey\bin\ffmpeg.exe

运行:
  python detect_hits_and_export_top10.py --video input.MOV --clear_output
"""

import os, shutil, argparse, subprocess, math
import numpy as np
import pandas as pd
import cv2, librosa
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

# =============== 默认参数（可被命令行覆盖） ===============
VIDEO_PATH   = "input.MOV"
FFMPEG_PATH  = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"

EXPORT_DIR   = "hit_top10_export"
AUDIO_FULL   = os.path.join(EXPORT_DIR, "audio_full.wav")
HITS_CSV     = os.path.join(EXPORT_DIR, "hits_list.csv")
AUDIO_CLIPS  = os.path.join(EXPORT_DIR, "audio_clips")
TOP10_DIR    = os.path.join(EXPORT_DIR, "top10_frames")
SUMMARY_CSV  = os.path.join(EXPORT_DIR, "top10_summary.csv")

CLEAR_OUTPUT = True
JPEG_QUALITY = 92

# 音频峰检测
SR           = 22050
HOP          = 512
SMOOTH_MED   = 7
THR_RATIO    = 0.20     # onset 相对阈值（最高值的比例）
PROM_RATIO   = 0.04    # prominence 相对阈值（<0 关闭）
MIN_SEP_SEC  = 2     # 邻峰最小间隔

# 音频/视频窗口（相对峰时刻的前后范围）
# 音频片段长度
CLIP_PRE     = 0
CLIP_POST    = 2
# 视频搜帧窗口长度（可以略比音频片段更宽）
WIN_PRE      = 0
WIN_POST     = 2.5
SKIP_FIRST_SEC = 0.4



# 视觉检测（HSV + 形态学开闭 + 圆度/面积阈值）
HSV_LOW      = (18,  80,  80)    # 可按实际球色微调
HSV_HIGH     = (65, 255, 255)
MORPH_K      = 5                 # 椭圆核尺寸（奇数）
OPEN_ITERS   = 1                 # 去小噪点
CLOSE_ITERS  = 2                 # 补裂缝/小孔
ROUNDNESS_MIN= 0.30              # 圆度下限（0~1）
AREA_MIN_FRAC= 1.5e-5            # 相对外接圆面积下限（相对整幅图）
AREA_MAX_FRAC= 0.30              # 相对外接圆面积上限（防止误检超大块）
GAUSS_BLUR   = True

TOPK         = 62           # 每次击球导出前K帧（球最大）

# =========================================================

def ensure_dir(p):
    if p and not os.path.exists(p): os.makedirs(p)

def run_ffmpeg(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def extract_audio_ffmpeg(video_path, audio_path, sr, ffmpeg_path):
    ensure_dir(os.path.dirname(audio_path))
    cmd = [ffmpeg_path, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), audio_path]
    print(">>> 提取整段音频…")
    run_ffmpeg(cmd)
    print(">>> 音频完成：", audio_path)

def onset_envelope(audio_path, sr, hop, smooth_med):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float64)
    if smooth_med and smooth_med > 1 and smooth_med % 2 == 1:
        onset = median_filter(onset, size=smooth_med)
    return onset

def detect_hits_from_onset(onset, sr, hop, thr_ratio, prom_ratio, min_sep_sec):
    if onset.size == 0: return np.array([])
    hop_sec = hop / sr
    min_dist = max(1, int(min_sep_sec / hop_sec))
    maxv = float(np.max(onset)) if onset.size else 1.0
    height = thr_ratio * maxv
    kwargs = {"height": height, "distance": min_dist}
    if prom_ratio is not None and prom_ratio >= 0:
        kwargs["prominence"] = prom_ratio * maxv
    peaks, _ = find_peaks(onset, **kwargs)
    return librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

def slice_audio_clip_ffmpeg(audio_full, ffmpeg_path, t0, t1, out_path):
    dur = max(0.01, t1 - t0)
    ensure_dir(os.path.dirname(out_path))
    cmd = [ffmpeg_path, "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", audio_full, out_path]
    run_ffmpeg(cmd)

def ball_size_fraction(bgr, hsv_low, hsv_high,
                       round_min, area_min_frac, area_max_frac,
                       morph_k=5, open_iters=1, close_iters=2, blur=True):
    """
    以"候选外接圆面积/整幅图面积"作为“球大小”评分（0~1），并返回圆度以供调试。
    """
    if bgr is None: return 0.0, 0.0
    if blur: bgr = cv2.GaussianBlur(bgr, (3,3), 0)
    H, W = bgr.shape[:2]
    total = float(H*W)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3,morph_k)|1, max(3,morph_k)|1))
    if open_iters > 0:  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=int(open_iters))
    if close_iters > 0: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(close_iters))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_frac = 0.0; best_circ = 0.0
    for c in cnts:
        A = float(cv2.contourArea(c))
        if A <= 1.0: continue
        P = float(cv2.arcLength(c, True)) or 1.0
        circ = 4.0*np.pi*A/(P*P)  # 0~1
        if circ < round_min: continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        size_frac = (np.pi*r*r) / total
        if area_min_frac <= size_frac <= area_max_frac and size_frac > best_frac:
            best_frac = float(size_frac)
            best_circ = float(circ)
    return best_frac, best_circ

def annotate_and_save(frame_bgr, save_path, frac, circ, time_sec):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    scale = max(0.6, min(1.6, (h/720.0)*0.9))
    thick = max(2, int(round(h/540.0)))
    text = f"size_frac={frac:.6f}  circ={circ:.2f}  t={time_sec:.3f}s"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 8
    cv2.rectangle(img, (5,5), (5+tw+2*pad, 5+th+2*pad), (0,0,0), -1)
    cv2.putText(img, text, (5+pad, 5+pad+th-4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), thick, cv2.LINE_AA)
    cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

def process_video_for_topk(video_path, peak_time, pre_s, post_s, topk,
                           hsv_low, hsv_high, round_min, amin, amax,
                           morph_k, open_iters, close_iters, blur,
                           out_dir):
    """
    在 [peak_time-pre_s, peak_time+post_s] 内枚举帧，计算“球大小”分(外接圆面积/全图面积)，
    先按 size_frac 选出 TopK，再按 time_sec 升序输出与保存。
    返回：list[ {frame, time_sec, size_frac, circularity, file} ]，按时间升序
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    start_sec = max(0.0, float(peak_time) - pre_s)
    end_sec   = min(duration, float(peak_time) + post_s)
    f0 = max(0, int(round(start_sec * fps)))
    f1 = min(total_frames - 1, int(round(end_sec   * fps)))

    scored = []
    for f in range(f0, f1 + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret: 
            continue
        t = f / fps
        frac, circ = ball_size_fraction(frame, hsv_low, hsv_high, round_min, amin, amax,
                                        morph_k, open_iters, close_iters, blur)
        if frac > 0:  # 仅保留非零得分
            scored.append((frac, circ, f, t, frame))

    cap.release()
    if not scored: 
        return []

    # 先按 frac 选 TopK
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[:max(1, topk)]

    # 再按 time 升序输出与保存
    keep.sort(key=lambda x: x[3])  # 按 time_sec 升序
    results = []
    for rank, (frac, circ, fidx, t, frame) in enumerate(keep, 1):
        fname = f"top{rank:02d}_f{fidx}_t{t:.3f}.jpg"
        savep = os.path.join(out_dir, fname)
        annotate_and_save(frame, savep, frac, circ, t)
        results.append({
            "frame": int(fidx),
            "time_sec": round(float(t), 6),
            "size_frac": round(float(frac), 6),
            "circularity": round(float(circ), 3),
            "file": fname
        })
    return results

def parse_args():
    ap = argparse.ArgumentParser(description="Audio-driven hit detection + per-hit audio clip + top-10 largest-ball frames")
    # 路径
    ap.add_argument("--video", default=VIDEO_PATH)
    ap.add_argument("--ffmpeg", default=FFMPEG_PATH)
    ap.add_argument("--export_dir", default=EXPORT_DIR)
    ap.add_argument("--audio_full", default=AUDIO_FULL)
    ap.add_argument("--audio_clips", default=AUDIO_CLIPS)
    ap.add_argument("--top10_dir", default=TOP10_DIR)
    ap.add_argument("--hits_csv", default=HITS_CSV)
    ap.add_argument("--summary_csv", default=SUMMARY_CSV)
    ap.add_argument("--clear_output", action="store_true", default=CLEAR_OUTPUT)

    # 音频峰检测
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--hop", type=int, default=HOP)
    ap.add_argument("--smooth_med", type=int, default=SMOOTH_MED)
    ap.add_argument("--thr", type=float, default=THR_RATIO)
    ap.add_argument("--prom", type=float, default=PROM_RATIO)
    ap.add_argument("--min_sep", type=float, default=MIN_SEP_SEC)

    # 窗口
    ap.add_argument("--clip_pre",  type=float, default=CLIP_PRE)
    ap.add_argument("--clip_post", type=float, default=CLIP_POST)
    ap.add_argument("--win_pre",   type=float, default=WIN_PRE)
    ap.add_argument("--win_post",  type=float, default=WIN_POST)

    # 视觉阈值
    ap.add_argument("--h_low",  type=int, nargs=3, default=list(HSV_LOW))
    ap.add_argument("--h_high", type=int, nargs=3, default=list(HSV_HIGH))
    ap.add_argument("--round_min", type=float, default=ROUNDNESS_MIN)
    ap.add_argument("--amin", type=float, default=AREA_MIN_FRAC)
    ap.add_argument("--amax", type=float, default=AREA_MAX_FRAC)
    ap.add_argument("--morph_k", type=int, default=MORPH_K)
    ap.add_argument("--open_iters", type=int, default=OPEN_ITERS)
    ap.add_argument("--close_iters", type=int, default=CLOSE_ITERS)
    ap.add_argument("--no_blur", action="store_true")

    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--jpeg_quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--audio_head_shift", type=float, default=0.5, help="为所有音频片段裁切整体右移（跳过）秒数，>=0")

    return ap.parse_args()

def main():
    args = parse_args()

    # 覆盖全局可视化参数
    global HSV_LOW, HSV_HIGH, GAUSS_BLUR, JPEG_QUALITY
    HSV_LOW  = tuple(args.h_low)
    HSV_HIGH = tuple(args.h_high)
    GAUSS_BLUR = not args.no_blur
    JPEG_QUALITY = args.jpeg_quality

    # 清理输出
    if args.clear_output and os.path.exists(args.export_dir):
        shutil.rmtree(args.export_dir)
    ensure_dir(args.export_dir)
    ensure_dir(args.audio_clips)
    ensure_dir(args.top10_dir)

    # 1) 抽取整段音频
    extract_audio_ffmpeg(args.video, args.audio_full, args.sr, args.ffmpeg)

    # 2) onset → 峰时刻
    onset = onset_envelope(args.audio_full, args.sr, args.hop, args.smooth_med)
    prom = None if (args.prom is not None and args.prom < 0) else args.prom
    hit_times = detect_hits_from_onset(onset, args.sr, args.hop, args.thr, prom, args.min_sep)

    # 写出峰列表
    pd.DataFrame({"hit_idx": list(range(len(hit_times))),
                  "peak_time_sec": [float(t) for t in hit_times]}).to_csv(args.hits_csv, index=False)
    print(f"✅ 已写出峰列表：{args.hits_csv}  (共 {len(hit_times)} 次击球)")

    # 打开视频获取时长，供安全裁剪
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError(f"无法打开视频：{args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    # 3) 循环每个击球：导出音频片段 + 视频窗口TopK
    rows = []
    pbar = tqdm(enumerate(hit_times), total=len(hit_times), desc="处理击球事件", unit="hit")
    for hit_idx, t in pbar:
        # 3.1 音频片段（对所有片段整体右移 audio_head_shift 秒）
        a0 = max(0.0, float(t) - args.clip_pre)
        a1 = float(t) + args.clip_post

        # 应用整体偏移（跳过文件开头若干秒），并做边界保护
        shift = max(0.0, float(args.audio_head_shift))
        eff_a0 = max(0.0, min(a0 + shift, duration))
        eff_a1 = max(eff_a0 + 0.01, min(a1 + shift, duration))  # 至少 10ms

        clip_path = os.path.join(args.audio_clips, f"hit_{hit_idx:04d}.wav")
        slice_audio_clip_ffmpeg(args.audio_full, args.ffmpeg, eff_a0, eff_a1, clip_path)


        # 3.2 视频窗口内TopK
        out_dir = os.path.join(args.top10_dir, f"hit_{hit_idx:04d}")
        topk_items = process_video_for_topk(
            args.video, t, args.win_pre, args.win_post, args.topk,
            HSV_LOW, HSV_HIGH, args.round_min, args.amin, args.amax,
            args.morph_k, args.open_iters, args.close_iters, GAUSS_BLUR,
            out_dir
        )

        if not topk_items:
            rows.append({"hit_idx": hit_idx, "peak_time_sec": float(t),
                         "audio_clip": os.path.relpath(clip_path, args.export_dir),
                         "rank": -1, "frame": -1, "time_sec": -1,
                         "size_frac": 0.0, "circularity": 0.0,
                         "file": ""})
        else:
            for rank, item in enumerate(topk_items, 1):
                rows.append({
                    "hit_idx": hit_idx,
                    "peak_time_sec": float(t),
                    "audio_clip": os.path.relpath(clip_path, args.export_dir),
                    "rank": rank,
                    "frame": int(item["frame"]),
                    "time_sec": float(item["time_sec"]),
                    "size_frac": float(item["size_frac"]),
                    "circularity": float(item["circularity"]),
                    "file": os.path.join(os.path.relpath(out_dir, args.export_dir), item["file"])
                })

    # 4) 汇总CSV
    pd.DataFrame(rows).to_csv(args.summary_csv, index=False)
    print(" 已写出汇总：", args.summary_csv)
    print(" Top10 目录：", os.path.abspath(args.top10_dir))
    print(" 音频片段：", os.path.abspath(args.audio_clips))

if __name__ == "__main__":
    main()
