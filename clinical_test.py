import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from tqdm import tqdm # 进度条库，如果没有请 pip install tqdm

# ================= 配置区域 =================
# 1. 路径设置
GT_VIDEO_PATH = r"E:\master\毕设\code\Polyp-PVT-main\result_map\video\GT\video_02_start_500.mp4"   # 人工标注Mask视频
PRED_VIDEO_PATH = r"E:\master\毕设\code\Polyp-PVT-main\result_map\video\AntrumPVT\video_02_start_500.mp4"   # 网络预测Mask视频

# 2. 视频参数
FPS = 29
PIXEL_TO_CM2 = 0.01  # 如果不知道具体比例，设为1，ACA和ACF计算不受影响，只有MI受影响

# ================= 核心处理函数 =================

def get_clean_area(mask):
    """
    关键函数：对网络输出的Mask进行清洗，只计算有效面积
    """
    # 转为单通道二值图
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # --- 步骤1: 形态学操作 (填补内部空洞，平滑边缘) ---
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # 闭运算填洞
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开运算去噪
    
    # --- 步骤2: 只保留最大连通域 (去除孤立噪点) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1: # 只有背景，没检测到东西
        return 0
    
    # stats[:, 4] 是面积 (第0个通常是背景，要排除)
    # 找到面积最大的连通域索引 (排除背景)
    max_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[max_label_idx, cv2.CC_STAT_AREA]
    
    return largest_area * PIXEL_TO_CM2

def process_video_to_curve(video_path, is_prediction=False):
    """
    读取视频并转换为面积曲线
    is_prediction: 如果是网络预测结果，开启强力清洗模式
    """
    cap = cv2.VideoCapture(video_path)
    areas = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用 tqdm 显示进度
    desc = "Processing Pred" if is_prediction else "Processing GT  "
    for _ in tqdm(range(total_frames), desc=desc):
        ret, frame = cap.read()
        if not ret: break
        
        if is_prediction:
            # 预测结果需要清洗
            area = get_clean_area(frame)
        else:
            # GT 默认是干净的，直接算
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            area = np.count_nonzero(frame) * PIXEL_TO_CM2
            
        areas.append(area)
        
    cap.release()
    return np.array(areas)

def calculate_metrics(areas, fps):
    """
    根据面积曲线计算 ACA, ACF, MI
    """
    # 1. 平滑处理 (Savitzky-Golay)
    # 窗口大小取 fps*2 左右，保证平滑掉毛刺
    window_length = int(fps * 2) 
    if window_length % 2 == 0: window_length += 1
    if len(areas) < window_length: window_length = len(areas) // 2 * 2 + 1
    
    smooth_areas = savgol_filter(areas, window_length, 3)
    
    # 2. 寻峰算法
    # 限制两个波峰之间至少间隔 1.5 秒
    min_dist = int(fps * 1.5)
    # 动态阈值：波动的 15% 作为最小突起高度，防止检测到平坦区域的微小波动
    dynamic_prominence = (np.max(smooth_areas) - np.min(smooth_areas)) * 0.15
    
    peaks, _ = find_peaks(smooth_areas, distance=min_dist, prominence=dynamic_prominence)
    valleys, _ = find_peaks(-smooth_areas, distance=min_dist, prominence=dynamic_prominence)
    
    # 3. 参数计算
    acas = []
    # 简单的配对策略：找每个波峰后最近的波谷
    for p in peaks:
        # 找该波峰之后的所有波谷
        future_valleys = valleys[valleys > p]
        if len(future_valleys) > 0:
            v = future_valleys[0] # 最近的一个
            s_relx = smooth_areas[p]
            s_cont = smooth_areas[v]
            
            # 避免除以0或负数
            if s_relx > 1e-5:
                aca = (s_relx - s_cont) / s_relx
                acas.append(aca)

    # 汇总
    mean_aca = np.mean(acas) * 100 if len(acas) > 0 else 0
    
    duration_min = len(areas) / fps / 60
    n_cont = len(acas)
    acf = n_cont / duration_min * 2 if duration_min > 0 else 0
    
    mi = mean_aca * acf
    
    return {
        "curve_smooth": smooth_areas,
        "curve_raw": areas,
        "peaks": peaks,
        "valleys": valleys,
        "ACA": mean_aca,
        "ACF": acf,
        "MI": mi
    }

# ================= 主程序 =================

# 1. 处理 GT 视频 (作为金标准)
print("正在处理 GT 视频...")
gt_areas = process_video_to_curve(GT_VIDEO_PATH, is_prediction=False)
gt_res = calculate_metrics(gt_areas, FPS)

# 2. 处理 预测 视频 (带有后处理清洗)
print("正在处理 预测 视频...")
pred_areas = process_video_to_curve(PRED_VIDEO_PATH, is_prediction=True)
pred_res = calculate_metrics(pred_areas, FPS)

# 3. 打印结果对比
print("\n" + "="*40)
print(f"{'Metric':<10} | {'GT (Gold Std)':<15} | {'Pred (Ours)':<15} | {'Error':<10}")
print("-" * 56)
for metric in ["ACA", "ACF", "MI"]:
    val_gt = gt_res[metric]
    val_pred = pred_res[metric]
    error = abs(val_pred - val_gt)
    print(f"{metric:<10} | {val_gt:<15.2f} | {val_pred:<15.2f} | {error:<10.2f}")
print("="*40)

# 4. 画图对比 (直接用于论文插图)
plt.figure(figsize=(12, 5))
times = np.arange(len(gt_areas)) / FPS

# 画 GT 曲线
plt.plot(times, gt_res['curve_smooth'], 'g-', label='Ground Truth (Smoothed)', linewidth=2, alpha=0.7)
# 画 预测 曲线
plt.plot(times, pred_res['curve_smooth'], 'r--', label='Antrum-PVT (Ours)', linewidth=2, alpha=0.8)

# 标记波峰 (可选)
# plt.plot(times[gt_res['peaks']], gt_res['curve_smooth'][gt_res['peaks']], 'go', markersize=5)
# plt.plot(times[pred_res['peaks']], pred_res['curve_smooth'][pred_res['peaks']], 'rx', markersize=5)

plt.title(f"Motility Assessment Comparison\nGT: ACA={gt_res['ACA']:.1f}, ACF={gt_res['ACF']:.1f} | Pred: ACA={pred_res['ACA']:.1f}, ACF={pred_res['ACF']:.1f}")
plt.xlabel("Time (s)")
plt.ylabel("CSA (Area)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()