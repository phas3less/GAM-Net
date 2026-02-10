import numpy as np
from PIL import Image
import os
import pandas as pd


def calculate_segmentation_metrics(y_true, y_pred):
    """
    计算二分类分割任务的Dice、MAE和IoU指标

    输入参数:
    y_true (numpy.ndarray): 真实标签的数组，shape为任意维度（如(H, W)或(N, H, W)）
    y_pred (numpy.ndarray): 模型预测结果的数组，shape需与y_true相同

    返回:
    tuple: (dice, mae, iou) 三个指标的浮点数值
    """
    # 展平数组并转换为float32类型以支持数学运算
    y_true_f = y_true.flatten().astype(np.float32)
    y_pred_f = y_pred.flatten().astype(np.float32)

    # 计算MAE（平均绝对误差）
    mae = np.mean(np.abs(y_true_f - y_pred_f))

    # 计算TP/FP/FN
    tp = np.sum(y_true_f * y_pred_f)
    fp = np.sum(y_pred_f) - tp
    fn = np.sum(y_true_f) - tp

    # 计算Dice系数
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    # 计算IoU（交并比）
    iou = tp / (tp + fp + fn + 1e-7)

    return dice, mae, iou







# 定义路径
pred_root = r"code\result_map\GAMNet"
gt_root = r"E:\datasets\used\UGASD_final\Annotations"
output_csv = r"E:\code\result_map\GAMNet\metrics_summary.csv"  # 结果保存路径

# 定义二值化阈值（根据预测结果调整）
THRESHOLD = 128


# 指标计算函数
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def jaccard_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union != 0 else 0.0


def mae_score(y_true, y_pred):
    # 将预测值归一化到 [0,1]，假设 y_pred 是 0~255
    return np.mean(np.abs(y_true - y_pred.astype(np.float32) / 255.0))

# 遍历所有视频序列文件夹
all_metrics = []
video_folders = [f for f in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, f))]

for folder in video_folders:
    pred_dir = os.path.join(pred_root, folder)
    gt_dir = os.path.join(gt_root, folder)  # 假设文件夹名与 gt 的子文件夹名一致

    # 如果 gt 没有子文件夹，直接使用 gt_root（需调整此处逻辑）
    # gt_dir = gt_root

    # 收集所有预测文件
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.bmp'))]

    # 逐文件处理
    video_metrics = []
    for pred_file in pred_files:
        # 读取预测图像并二值化
        pred_path = os.path.join(pred_dir, pred_file)
        # pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取
        pred = np.array(Image.open(pred_path))
        pred_bin = (pred > THRESHOLD).astype(np.uint8)

        # 读取对应的真实标签（假设文件名与 pred_file 相同）
        gt_path = os.path.join(gt_dir, pred_file)  # 如果扩展名不同，需处理文件名
        # 如果 gt 是 .png 格式而 pred 是 .jpg：
        # gt_path = os.path.join(gt_dir, os.path.splitext(pred_file)[0] + ".png")

        if not os.path.exists(gt_path):
            print(f"Warning: {gt_path} not found, skipping.")
            continue

        gt_bin = np.array(Image.open(gt_path))
        # gt_bin = (gt > 0).astype(np.uint8)  # 假设 gt 是 0/255 的二值图

        # 检查尺寸一致性
        if pred_bin.shape != gt_bin.shape:
            print(f"Shape mismatch in {pred_file}, skipping.")
            continue

        # 计算指标
        dice,mae,iou = calculate_segmentation_metrics(gt_bin, pred_bin)

        video_metrics.append({
            "sequence": folder,
            "file": pred_file,
            "Dice": dice,
            "IOU": iou,
            "MAE": mae
        })


    # 计算当前视频序列的平均值
    if video_metrics:
        df_video = pd.DataFrame(video_metrics)
        mean_metrics = df_video.mean(numeric_only=True).to_dict()
        mean_metrics["sequence"] = folder
        mean_metrics["file"] = "Average"
        all_metrics.append(mean_metrics)

# 汇总所有结果并保存
if all_metrics:
    df = pd.DataFrame(all_metrics)
    # 添加整体统计
    overall_mean = df.mean(numeric_only=True)
    overall_std = df.std(numeric_only=True)
    df = pd.concat([
        df,
        pd.DataFrame([{**{"sequence": "OVERALL", "file": "Mean"}, **overall_mean.to_dict()}]),
        pd.DataFrame([{**{"sequence": "OVERALL", "file": "Std"}, **overall_std.to_dict()}])
    ], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
else:

    print("No valid data processed.")
