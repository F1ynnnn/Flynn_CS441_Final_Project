#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 good_frame 中的图片 -> 映射到 top10_frames/ hit_XXXX 事件 -> 拷贝 audio_clips 中对应音频
输出到 good_frame_audio/ 并重命名为该帧号（fXXXX.*），同时写出映射 CSV。
"""

import re
import csv
import shutil
from pathlib import Path

# ===== 手动设置：按你的目录结构 =====
ROOT              = Path(__file__).resolve().parent
GOOD_FRAMES_DIR   = ROOT / "good_frame"
TOP10_FRAMES_DIR  = ROOT / "hit_top10_export" / "top10_frames"   # 有 hit_xxxx 子文件夹
AUDIO_CLIPS_DIR   = ROOT / "hit_top10_export" / "audio_clips"
OUT_DIR           = ROOT / "good_frame_audio"
MAP_CSV           = ROOT / "good_frame_audio_map.csv"
# ====================================

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

# 提取帧号（尽量宽松）：f1843 / frame_1843 / …_1843 / …-1843
FRAME_PATTERNS = [
    re.compile(r"f(?P<frame>\d{3,7})", re.IGNORECASE),
    re.compile(r"frame[_-]?(?P<frame>\d{3,7})", re.IGNORECASE),
    re.compile(r"(?<!\d)(?P<frame>\d{3,7})(?!\d)"),  # 孤立数字段
]

def get_frame_num_from_name(name: str):
    for pat in FRAME_PATTERNS:
        m = pat.search(name)
        if m:
            return m.group("frame")
    return None

def build_frame_to_event_index(top10_dir: Path):
    """
    扫描 hit_top10_export/top10_frames 下的所有 hit_xxxx 子目录，
    遍历里面的文件名，解析出帧号，建立映射： frame_num -> event_id('0013' 这种)。
    """
    index = {}
    hit_dirs = sorted([d for d in top10_dir.iterdir() if d.is_dir() and d.name.lower().startswith("hit_")])
    for d in hit_dirs:
        # 事件编号（保留零填充）
        m = re.match(r"hit[_-]?(?P<eid>\d+)$", d.name, flags=re.IGNORECASE)
        if not m:
            continue
        eid = m.group("eid")
        # 遍历该事件目录下所有图像文件
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue
            fnum = get_frame_num_from_name(p.stem) or get_frame_num_from_name(p.name)
            if fnum:
                # 一个帧号可能被多个事件包含；一般不会发生。若发生，保留先遇到的并记录冲突。
                if fnum not in index:
                    index[fnum] = eid
    return index

def find_audio_for_event(audio_dir: Path, event_id: str):
    """
    在 audio_clips 中查找包含 event_id 的音频文件。
    优先精确匹配 'hit_<id>'，否则退化为包含 '<id>'.
    """
    candidates = []
    for p in audio_dir.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            nm = p.name.lower()
            if f"hit_{event_id}".lower() in nm or f"hit-{event_id}".lower() in nm:
                candidates.append(p)
            elif event_id in nm:
                candidates.append(p)
    # 去重并排序，取第一个
    if not candidates:
        return None
    candidates = sorted(set(candidates))
    return candidates[0]

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    # 基础检查
    for d in [GOOD_FRAMES_DIR, TOP10_FRAMES_DIR, AUDIO_CLIPS_DIR]:
        if not d.exists():
            raise FileNotFoundError(f"找不到目录：{d}")

    ensure_outdir(OUT_DIR)

    # 1) 建立索引：帧号 -> 事件编号
    print("📇 正在构建帧号→事件编号索引……")
    frame2event = build_frame_to_event_index(TOP10_FRAMES_DIR)
    print(f"  索引大小：{len(frame2event)}")

    # 2) 遍历 good_frame 中的图片
    images = sorted([p for p in GOOD_FRAMES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        print("⚠️ good_frame 目录里没有图片。")
        return

    rows = []
    ok, miss_frame, miss_event, miss_audio = 0, 0, 0, 0

    for img in images:
        fname = img.name
        fnum = get_frame_num_from_name(img.stem) or get_frame_num_from_name(img.name)
        if not fnum:
            print(f"⚠️ 无法解析帧号：{fname}")
            rows.append([fname, "", "", "", "no_frame_number"])
            miss_frame += 1
            continue

        eid = frame2event.get(fnum)
        if not eid:
            print(f"⚠️ 未在 top10_frames 索引中找到帧 {fnum} 的事件编号")
            rows.append([fname, fnum, "", "", "event_not_found"])
            miss_event += 1
            continue

        audio_file = find_audio_for_event(AUDIO_CLIPS_DIR, eid)
        if not audio_file:
            print(f"⚠️ 未在 audio_clips 找到事件 {eid} 的音频")
            rows.append([fname, fnum, eid, "", "audio_not_found"])
            miss_audio += 1
            continue

        # 复制并按帧号重命名
        out_audio = OUT_DIR / f"f{fnum}{audio_file.suffix.lower()}"
        try:
            shutil.copy2(audio_file, out_audio)
            print(f"✅ {fname}  ->  事件 {eid}  ->  {out_audio.name}")
            rows.append([fname, fnum, eid, str(out_audio), "ok"])
            ok += 1
        except Exception as e:
            print(f"❌ 复制失败：{audio_file} -> {out_audio}：{e}")
            rows.append([fname, fnum, eid, str(audio_file), f"copy_failed:{e}"])

    # 3) 写汇总 CSV
    with open(MAP_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["good_frame_file", "frame_number", "event_id", "out_audio_path", "status"])
        wr.writerows(rows)

    print("\n==== 汇总 ====")
    print(f"成功复制音频：{ok}")
    print(f"无法解析帧号：{miss_frame}")
    print(f"未找到事件编号：{miss_event}")
    print(f"未找到音频：{miss_audio}")
    print(f"映射表：{MAP_CSV}")

if __name__ == "__main__":
    main()
