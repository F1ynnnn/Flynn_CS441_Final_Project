#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量五点标注（tip, throat_mid, p3, p9, ball）+ 模板掩膜投影 + 导出结果到新文件夹

操作:
  左键：按顺序点击点
  中键：撤销（先撤球点，再撤关键点）
  右键：warp（只需要前4点）
  S：   保存当前图片全部产物并自动进入下一张
  N：   跳过当前图片（不保存）进入下一张
  R：   重置当前图片的点
  Q/ESC：退出（进度保留，已保存的不会丢）
"""

import cv2, json, numpy as np, pandas as pd
from pathlib import Path
import os

# ========= 1) 手动设置 =========
INPUT_DIR       = "good_frame"        # 待标注图片文件夹
OUTPUT_DIR      = "good_frame_tags"   # 输出结果文件夹（自动创建）
TEMPLATE_JSON   = "template_meta.json"
TEMPLATE_MASK   = "template_mask.png"
EXPORT_CONTOUR_CSV = True             # 导出等弧长360点
SKIP_EXISTING      = True             # 若目标文件已存在则跳过
# =================================

# ---- Unicode安全的图像读写 ----
def imread_u8(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_u8(path, img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("cv2.imencode failed")
    buf.tofile(str(path))

# ---- 载入模板 ----
root = Path(__file__).resolve().parent
in_dir  = (root / INPUT_DIR)
out_dir = (root / OUTPUT_DIR)
tpl_json = root / TEMPLATE_JSON
tpl_mask = root / TEMPLATE_MASK

if not in_dir.is_dir():
    raise FileNotFoundError(f"找不到输入文件夹: {in_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

with open(tpl_json, "r", encoding="utf-8") as f:
    meta = json.load(f)
tpl = imread_u8(tpl_mask, cv2.IMREAD_GRAYSCALE)
if tpl is None:
    raise FileNotFoundError(f"无法读取模板掩膜: {tpl_mask}")

kps0 = meta["keypoints"]
src_pts = np.float32([kps0["tip"], kps0["throat_mid"], kps0["p3"], kps0["p9"]])

ORDER = ["tip", "throat_mid", "p3", "p9", "ball"]

# ---- 收集所有图片 ----
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

if not images:
    raise SystemExit(f"{in_dir} 里没有图片")

idx = 0
win = "batch_mark_5pts"

def process_one(img_path):
    """交互式标注一张图，保存产物。按 S 保存并返回 True; N 跳过返回 True; Q 返回 False 结束全流程。"""
    img = imread_u8(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ 无法读取图片：{img_path}")
        return True  # 继续下一张

    H, W = img.shape[:2]
    stem = img_path.stem
    # 输出文件路径
    out_subdir = out_dir / stem
    out_subdir.mkdir(parents=True, exist_ok=True)
    f_mask    = out_subdir / f"{stem}_mask.png"
    f_overlay = out_subdir / f"{stem}_overlay.png"
    f_points  = out_subdir / f"{stem}_points.json"
    f_ballcsv = out_subdir / f"{stem}_ball.csv"
    f_contour = out_subdir / f"{stem}_contour360.csv"

    if SKIP_EXISTING and f_points.exists() and f_mask.exists():
        print(f"⏭️ 已存在，跳过：{img_path.name}")
        return True

    dst_pts = []   # 四关键点
    ball_pt = None
    overlay, warped = None, None

    def draw_ui():
        base = img if overlay is None else overlay
        disp = base.copy()
        cv2.putText(disp, f"[{idx+1}/{len(images)}] {img_path.name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(disp, "L:add  M:undo  R:warp  S:save  N:skip  Rst:reset  Q:quit",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        filled = len(dst_pts) + (1 if ball_pt is not None else 0)
        if filled < 5:
            cv2.putText(disp, f"Now: {ORDER[filled]}", (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 四关键点
        for i, p in enumerate(dst_pts):
            cv2.circle(disp, p, 6, (0,255,255), -1)
            cv2.putText(disp, ORDER[i], (p[0]+8, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # 球点
        if ball_pt is not None:
            cv2.circle(disp, ball_pt, 7, (0,0,255), -1)
            cv2.putText(disp, "ball", (ball_pt[0]+8, ball_pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return disp

    def on_mouse(event, x, y, flags, param):
        nonlocal dst_pts, ball_pt, overlay, warped
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(dst_pts) < 4:
                dst_pts.append((x, y))
            elif ball_pt is None:
                ball_pt = (x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            if ball_pt is not None:
                ball_pt = None; overlay = None; warped = None
            elif dst_pts:
                dst_pts.pop(); overlay = None; warped = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(dst_pts) < 4:
                print("⚠️ 需要先点击四关键点 (tip, throat_mid, p3, p9)")
                return
            dst = np.float32(dst_pts)
            Hmat, _ = cv2.findHomography(src_pts, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if Hmat is None:
                print("❌ Homography失败，检查点位")
                overlay = None; warped = None; return
            warped = cv2.warpPerspective(tpl, Hmat, (W, H), flags=cv2.INTER_NEAREST)
            overlay = img.copy()
            overlay[warped>0] = (0.4*overlay[warped>0] + 0.6*np.array([0,255,0])).astype(np.uint8)
            if ball_pt is not None:
                cv2.circle(overlay, ball_pt, 7, (0,0,255), -1)
                cv2.putText(overlay, "ball", (ball_pt[0]+8, ball_pt[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            print("✅ warp完成，按 S 保存")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        ui = draw_ui()
        cv2.imshow(win, ui)
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord('q'), ord('Q')):          # 退出整个批次
            cv2.destroyWindow(win)
            return False
        elif k in (ord('n'), ord('N')):            # 跳过这一张
            print("⏭️ 跳过本张")
            cv2.destroyWindow(win)
            return True
        elif k in (ord('r'), ord('R')):            # 重置本张
            dst_pts.clear(); ball_pt = None; overlay = None; warped = None
            print("↺ 重置，请重新点击")
        elif k in (ord('s'), ord('S')):            # 保存本张
            if warped is None:
                print("⚠️ 先右键warp再保存")
                continue
            # 保存图像
            imwrite_u8(f_mask, warped, ".png")
            imwrite_u8(f_overlay, overlay if overlay is not None else img, ".png")
            # 保存点
            payload = {}
            for i, name in enumerate(["tip","throat_mid","p3","p9"]):
                payload[name] = [int(dst_pts[i][0]), int(dst_pts[i][1])] if i < len(dst_pts) else None
            payload["ball"] = [int(ball_pt[0]), int(ball_pt[1])] if ball_pt is not None else None
            with open(f_points, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # 保存球点CSV
            if ball_pt is not None:
                pd.DataFrame([[img_path.name, ball_pt[0], ball_pt[1]]],
                             columns=["frame","ball_x","ball_y"]).to_csv(f_ballcsv, index=False)
            # 导出轮廓360点
            if EXPORT_CONTOUR_CSV:
                cnts, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    eps = 0.0025 * cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, eps, True)
                    pts = cnt.reshape(-1,2).astype(np.float32)
                    if not np.allclose(pts[0], pts[-1]):
                        pts = np.vstack([pts, pts[0]])
                    seg = pts[1:] - pts[:-1]
                    d = np.hypot(seg[:,0], seg[:,1])
                    s = np.hstack([[0], np.cumsum(d)])
                    L = s[-1]
                    t = np.linspace(0, L, 360, endpoint=False)
                    res, j = [], 0
                    for ti in t:
                        while s[j+1] < ti and j < len(d)-1:
                            j += 1
                        a = (ti - s[j]) / (d[j] + 1e-9)
                        res.append((pts[j] + a*(pts[j+1]-pts[j])).tolist())
                    pd.DataFrame(res, columns=["x","y"]).to_csv(f_contour, index=False)

            print(f"✅ 已保存到：{out_subdir}")
            cv2.destroyWindow(win)
            return True

# ---------- 主循环 ----------
while idx < len(images):
    keep = process_one(images[idx])
    if keep is False:  # 用户退出
        break
    idx += 1

print("🎉 批量标注结束。")
