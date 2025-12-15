#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ExtraTreesRegressor on peak-aligned raw waveform (fixed-length)

目录结构（工作目录：使用声音的击球点位置分析）：
    normalized_results_1/
    normalized_results_2/
    ...
    normalized_results_5/
    data/
        good_frame_audio_1/
        good_frame_audio_2/
        ...
        good_frame_audio_5/

匹配规则（严格按编号）：
    good_frame_audio_i 只匹配 normalized_results_i
    fXXXX.wav → 查找 normalized_results_i 中包含 fXXXX 的 CSV

输出：
    测试集 MSE / MAE / 半径误差（归一化坐标）
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import librosa

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----------------------------
# 1) 峰值对齐裁剪
# -----------------------------

def crop_around_peak(y, sr, pre_ms=40.0, post_ms=80.0, rms_frame=1024, rms_hop=256):
    """以 RMS 能量峰值为中心裁剪波形窗口（对齐撞击瞬态）"""
    if y is None or y.size == 0:
        return y

    rms = librosa.feature.rms(y=y, frame_length=rms_frame, hop_length=rms_hop, center=True)[0]
    peak_frame = int(np.argmax(rms))
    peak_sample = peak_frame * rms_hop

    pre = int(sr * (pre_ms / 1000.0))
    post = int(sr * (post_ms / 1000.0))

    start = max(0, peak_sample - pre)
    end = min(len(y), peak_sample + post)

    y_crop = y[start:end]
    target_len = pre + post
    if len(y_crop) < target_len:
        y_crop = np.pad(y_crop, (0, target_len - len(y_crop)), mode="constant")

    return y_crop


# -----------------------------
# 2) raw waveform 特征（定长向量）
# -----------------------------

def extract_raw_waveform(
    wav_path,
    sr=22050,
    peak_align=True,
    pre_ms=40.0,
    post_ms=80.0,
    duration_sec=0.12,     # peak_align=False 时用
):
    """
    返回：定长 raw waveform 向量 shape=(L,)
    - peak_align=True：用 pre_ms+post_ms 定义窗口长度
    - peak_align=False：用 duration_sec 定义窗口长度
    """
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    if len(y) < int(sr * 0.02):
        fixed_len = int(sr * duration_sec) if not peak_align else int(sr * ((pre_ms + post_ms) / 1000.0))
        return np.zeros(fixed_len, dtype=np.float32)

    if peak_align:
        y = crop_around_peak(y, sr, pre_ms=pre_ms, post_ms=post_ms, rms_frame=1024, rms_hop=256)
        fixed_len = len(y)
    else:
        fixed_len = int(sr * duration_sec)

    # 幅值归一化（减轻录音增益差异）
    mx = np.max(np.abs(y)) + 1e-9
    y = y / mx

    # pad/crop 到 fixed_len
    if len(y) < fixed_len:
        y = np.pad(y, (0, fixed_len - len(y)), mode="constant")
    elif len(y) > fixed_len:
        y = y[:fixed_len]

    return y.astype(np.float32)


# -----------------------------
# 3) CSV 匹配与标签读取
# -----------------------------

def find_matching_csv(audio_id, csv_dir):
    """仅在指定 csv_dir 内匹配包含 audio_id 的 CSV"""
    for csv_path in glob.glob(os.path.join(csv_dir, "*.csv")):
        if audio_id in os.path.basename(csv_path):
            return csv_path
    return None


def load_target_from_csv(csv_path):
    """返回标签 (x_norm, y_norm)；若只有极坐标则转换为 xy"""
    df = pd.read_csv(csv_path)

    if "x_norm" in df.columns and "y_norm" in df.columns:
        xs = df["x_norm"].astype(float).to_numpy()
        ys = df["y_norm"].astype(float).to_numpy()
    elif "r_norm" in df.columns and "theta" in df.columns:
        r = df["r_norm"].astype(float).to_numpy()
        th = df["theta"].astype(float).to_numpy()
        xs = r * np.cos(th)
        ys = r * np.sin(th)
    else:
        raise ValueError(f"{csv_path} 缺少坐标列 (x_norm,y_norm) 或 (r_norm,theta)")

    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


# -----------------------------
# 4) 构建数据集（严格按编号）
# -----------------------------

def build_dataset():
    X_list, y_list = [], []
    total_audio = 0
    paired = 0

    for i in range(1, 6):
        audio_dir = os.path.join("data", f"good_frame_audio_{i}")
        csv_dir = f"normalized_results_{i}"

        if not os.path.isdir(audio_dir):
            warnings.warn(f"未找到音频目录：{audio_dir}")
            continue
        if not os.path.isdir(csv_dir):
            warnings.warn(f"未找到 CSV 目录：{csv_dir}")
            continue

        wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        print(f"[组 {i}] 音频数量：{len(wavs)}")

        for wav_path in wavs:
            total_audio += 1
            audio_id = os.path.splitext(os.path.basename(wav_path))[0]

            csv_path = find_matching_csv(audio_id, csv_dir)
            if csv_path is None:
                continue

            try:
                feat = extract_raw_waveform(
                    wav_path,
                    sr=22050,
                    peak_align=True,
                    pre_ms=40.0,
                    post_ms=80.0,
                )
                target = load_target_from_csv(csv_path)
            except Exception as e:
                warnings.warn(f"处理失败：{wav_path} / {csv_path} | {e}")
                continue

            X_list.append(feat)
            y_list.append(target)
            paired += 1

    if paired == 0:
        raise RuntimeError("成功配对为 0：请检查 wav 与 csv 命名/路径规则。")

    X = np.stack(X_list)  # (N, L)
    y = np.stack(y_list)  # (N, 2)

    print(f"\n总音频数：{total_audio}")
    print(f"成功配对：{paired}")
    print(f"raw 波形维度：{X.shape[1]}\n")

    return X, y


# -----------------------------
# 5) 训练 ExtraTrees 并评估
# -----------------------------

def train_extratrees(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集：{X_train.shape[0]}      测试集：{X_test.shape[0]}")

    model = ExtraTreesRegressor(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.2,      # 高维 raw 常比 sqrt/log2 更好
        bootstrap=False,       # ExtraTrees 通常不需要 bootstrap
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    radial_err = np.linalg.norm(y_test - pred, axis=1).mean()

    print("\n===== ExtraTrees (raw waveform) 测试集表现 =====")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"半径误差（归一化坐标）: {radial_err:.6f}")

    return model


def main():
    X, y = build_dataset()
    train_extratrees(X, y)


if __name__ == "__main__":
    main()
