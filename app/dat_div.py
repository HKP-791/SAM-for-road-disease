import os
import numpy as np
from PIL import Image
import cv2
import random
import shutil

def generate_index(folder_path):
    # 获取文件夹中的所有照片文件
    image_files = [f.split('.', 1)[0] for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
    return image_files

def norm_gry_label(label):
    if len(label.shape)==3:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    nobel = np.max(label)
    return label/nobel

def process_images_and_labels(image_files, folder_path, label_dir, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
 # 创建或打开一个txt文件来存储生成的npz文件名
    index_file_path = os.path.join(output_folder, 'index.txt')
    with open(index_file_path, 'a') as index_file:
        for image_file in image_files:
            # 读取图像文件
            try:
                image_path = os.path.join(folder_path, f'{image_file}.png')
                image = np.array(Image.open(image_path)) / 255.0
            except:
                try:
                    image_path = os.path.join(folder_path, f'{image_file}.jpg')
                    image = np.array(Image.open(image_path)) / 255.0
                except:
                    continue

            try:
                label_path = os.path.join(label_dir, f'{image_file}.png')
                label = np.array(Image.open(label_path))
                label = norm_gry_label(label)
            except:
                try:
                    label_path = os.path.join(label_dir, f'{image_file}.jpg')
                    label = np.array(Image.open(label_path))
                    label = norm_gry_label(label)
                except:
                    continue

            # 将图像和标签存储为字典
            sample = {'image': image, 'label': label}
            
            # 保存为 .npz 文件
            idx = len(os.listdir(output_folder))
            output_file = os.path.join(output_folder, f'crack{idx:04}.npz')
            np.savez(output_file, **sample)
            
            # 将生成的npz文件名写入txt文件
            index_file.write(f'crack{idx:04}\n')

def random_move_files(source_folder, destination_folder, ratio, log_file, src_file):
    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    num_files = int(len(files) * ratio)
    
    # 随机选择指定数量的文件
    selected_files = random.sample(files, num_files)
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 移动选中的文件到目标文件夹
    for file in selected_files:
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.move(source_file, destination_file)
    
    # 将选中的文件名写入日志文件
    with open(log_file, 'w') as f:
        for file in selected_files:
            f.write(file + '\n')

    content_to_remove = file
    # 读取文件内容
    with open(src_file, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    # 删除指定内容
    modified_contents = file_contents.replace(content_to_remove, '')
    # 将修改后的内容写回文件
    with open(src_file, 'w', encoding='utf-8') as file:
        file.write(modified_contents)

# # 使用示例
# source_folder = r'X:\SAMed_fortest\train\train_npz'
# destination_folder = r'X:\SAMed_fortest\testset\test_vol_h5'
# log_file = r'X:\SAMed_fortest\lists\lists_Synapse\test_vol.txt'
# src_file = r'X:\SAMed_fortest\lists\lists_Synapse\train.txt'
# random_move_files(source_folder, destination_folder, 0.1, log_file, src_file)

# 示例使用
# folder_path = r'X:\Transportation_infrastructure_defect_dataset\pavement\GAPs384\VOC2012\raw_images'
# label_dir = r'X:\Transportation_infrastructure_defect_dataset\pavement\GAPs384\VOC2012\raw_masks'
# output_folder = r'X:\Transportation_infrastructure_defect_dataset\dataset'
# image_files = generate_index(folder_path)
# process_images_and_labels(image_files, folder_path, label_dir,  output_folder)

list_dir = os.listdir(r'X:\SAMed_fortest\testset\test_vol_h5')
with open(r'X:\SAMed_fortest\1.txt', 'w', encoding='utf-8') as f:
    for file in list_dir:
        f.write(file + '\n')