import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from collections import defaultdict

def calculate_metrics(true_mask, pred_mask):
    """
    计算PA、CPA、IoU、Dice系数和HD95
    """
    # 确保输入是二值图像
    true_mask = true_mask > 0
    pred_mask = pred_mask > 0

    # PA（像素准确率）
    pa = np.mean(true_mask == pred_mask)

    # CPA（类别像素准确率）
    tp = np.sum((true_mask == 1) & (pred_mask == 1))
    fp = np.sum((true_mask == 0) & (pred_mask == 1))
    fn = np.sum((true_mask == 1) & (pred_mask == 0))
    cpa = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    # IoU（交并比）
    intersection = np.sum((true_mask == 1) & (pred_mask == 1))
    union = np.sum((true_mask == 1) | (pred_mask == 1))
    iou = intersection / union if union != 0 else 0

    # Dice系数
    dice = 2 * intersection / (np.sum(true_mask) + np.sum(pred_mask)) if (np.sum(true_mask) + np.sum(pred_mask)) != 0 else 0

    # HD95（Hausdorff距离的95百分位）
    hd95 = max(directed_hausdorff(true_mask, pred_mask)[0], directed_hausdorff(pred_mask, true_mask)[0])

    return pa, cpa, iou, dice, hd95

def validate_image_format(image_path):
    """
    验证图像格式是否为要求的格式
    """
    valid_formats = ('.jpg', '.png', '.bmp', '.npy')
    return os.path.splitext(image_path)[1].lower() in valid_formats

def load_image(image_path):
    """
    根据文件格式加载图像
    """
    if image_path.endswith('.npy'):
        return np.load(image_path)
    else:
        return np.array(Image.open(image_path).convert('L'))

def main(true_mask_folder, pred_mask_folder):
    # 检查文件夹是否存在
    if not os.path.exists(true_mask_folder) or not os.path.exists(pred_mask_folder):
        print("指定的文件夹不存在，请检查路径！")
        return

    # 获取文件夹中的文件名
    true_mask_files = set(os.listdir(true_mask_folder))
    pred_mask_files = set(os.listdir(pred_mask_folder))

    # 检查文件格式和命名匹配
    for folder, files in zip([true_mask_folder, pred_mask_folder], [true_mask_files, pred_mask_files]):
        for file in files:
            if not validate_image_format(file):
                print(f"请上传jpg、png、bmp或npy格式照片，文件 {os.path.join(folder, file)} 格式不正确！")
                return

    if true_mask_files != pred_mask_files:
        print("出现了命名不匹配的照片，请检查后上传！")
        return

    # 初始化结果存储
    metrics = defaultdict(list)

    # 遍历文件夹中的文件
    for file in true_mask_files:
        true_mask_path = os.path.join(true_mask_folder, file)
        pred_mask_path = os.path.join(pred_mask_folder, file)

        # 加载图像
        true_mask = load_image(true_mask_path)
        pred_mask = load_image(pred_mask_path)

        # 计算指标
        pa, cpa, iou, dice, hd95 = calculate_metrics(true_mask, pred_mask)
        metrics['PA'].append(pa)
        metrics['CPA'].append(cpa)
        metrics['IoU'].append(iou)
        metrics['Dice'].append(dice)
        metrics['HD95'].append(hd95)

        print(f"文件 {file} 的指标：PA={pa:.4f}, CPA={cpa:.4f}, IoU={iou:.4f}, Dice={dice:.4f}, HD95={hd95:.4f}")

    # 计算均值
    print("\n所有照片的平均指标：")
    for metric, values in metrics.items():
        print(f"{metric} 均值: {np.mean(values):.4f}")

if __name__ == "__main__":
    true_mask_folder = input("请输入真值掩码图像文件夹路径：")
    pred_mask_folder = input("请输入模型预测掩码图像文件夹路径：")
    main(true_mask_folder, pred_mask_folder)