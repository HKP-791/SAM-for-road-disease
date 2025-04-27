import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
from PIL import Image
import shutil
import random
import matplotlib.pyplot as plt


def RDD_prcess(xml_path, img_dir):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_name = root.find('filename').text
    image_path = os.path.join(img_dir, f'{img_name}')
    image = np.array(Image.open(image_path))
    raw = image
    image = image / np.max(image)

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    mask = np.zeros((height, width), dtype=float)

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        if name == 'crack':
            mask[ymin:ymax, xmin:xmax] = 1.0
        elif name == 'pothole':
            mask[ymin:ymax, xmin:xmax] = 2.0
        elif name == 'patch':
            mask[ymin:ymax, xmin:xmax] = 3.0

    return raw, mask

def RDD_dat(xml_dir, img_dir, output_dir, index_path, jpg_dir):
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            image, label = RDD_prcess(os.path.join(xml_dir, file), img_dir)
        sample = {'image': image, 'label': label}

        idx = len(os.listdir(output_dir)) + 1
        output_file = os.path.join(output_dir, f'crack{idx:04}.png')
        jpg_file = os.path.join(jpg_dir, f'crack{idx:04}.jpg')
        # np.savez(output_file, **sample)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_file, label.astype(np.uint8))
        cv2.imwrite(jpg_file, image)

        with open(index_path, 'a') as index_file:
            index_file.write(f'crack{idx:04}\n')

def generate_index(folder_path):
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
    
    index_file_path = os.path.join(output_folder, 'index.txt')
    with open(index_file_path, 'a') as index_file:
        for image_file in image_files:
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

            sample = {'image': image, 'label': label}
            
            idx = len(os.listdir(output_folder))
            output_file = os.path.join(output_folder, f'crack{idx:04}.npz')
            np.savez(output_file, **sample)
            
            index_file.write(f'crack{idx:04}\n')

def random_move_files(source_folder, destination_folder, ratio, log_file, src_file):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    num_files = int(len(files) * ratio)
    
    selected_files = random.sample(files, num_files)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for file in selected_files:
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.move(source_file, destination_file)
    
    with open(log_file, 'w') as f:
        for file in selected_files:
            f.write(file + '\n')

    content_to_remove = file
    with open(src_file, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    modified_contents = file_contents.replace(content_to_remove, '')
    with open(src_file, 'w', encoding='utf-8') as file:
        file.write(modified_contents)

def del_text(src, dir):
    with open(src, 'r', encoding='utf-8') as file1:
        content_to_remove = file1.read().splitlines()

    with open(dir, 'r', encoding='utf-8') as file2:
        original_lines = file2.read().splitlines()

    filtered_lines = []
    for line in original_lines:
        if line not in content_to_remove:
            filtered_lines.append(line)

    with open(dir, 'w', encoding='utf-8') as file2_filtered:
        for line in filtered_lines:
            file2_filtered.write(line + '\n')

def random_delete_90_percent(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    num_lines_to_delete = int(len(lines) * 0.9)
    lines_to_delete = random.sample(lines, num_lines_to_delete)
    lines_to_delete_set = set(lines_to_delete)
    remaining_lines = [line for line in lines if line not in lines_to_delete_set]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(remaining_lines)

    print(f"已从 {input_file_path} 中删除了 {num_lines_to_delete} 行，并将剩余的 {len(remaining_lines)} 行写入 {output_file_path}")

if __name__ == '__main__':
    # source_folder = r'X:\Transportation_infrastructure_defect_dataset\RDD_train'
    # destination_folder = r'X:\Transportation_infrastructure_defect_dataset\RDD_test'
    # log_file = r'X:\Transportation_infrastructure_defect_dataset\test_vol.txt'
    # src_file = r'X:\Transportation_infrastructure_defect_dataset\train.txt'
    # random_move_files(source_folder, destination_folder, 0.1, log_file, src_file)

    # src = r'X:\Transportation_infrastructure_defect_dataset\val.txt'
    # dir = r'X:\Transportation_infrastructure_defect_dataset\train.txt'
    # del_text(src, dir)

    # folder_path = r'X:\Transportation_infrastructure_defect_dataset\pavement\CrackTree\VOC2012\JPEGImages'
    # label_dir = r'X:\Transportation_infrastructure_defect_dataset\pavement\CrackTree\VOC2012\SegmentationClass'
    # output_folder = r'X:\Transportation_infrastructure_defect_dataset\dat4_train'
    # image_files = generate_index(folder_path)
    # process_images_and_labels(image_files, folder_path, label_dir,  output_folder)

    xml_dir = r'X:\Transportation_infrastructure_defect_dataset\pavement\RDD2022_CN\VOC2012\Annotations'
    img_dir = r'X:\Transportation_infrastructure_defect_dataset\pavement\RDD2022_CN\VOC2012\JPEGImages'
    output_dir = r'X:\Transportation_infrastructure_defect_dataset\data'
    index_path = r'X:\Transportation_infrastructure_defect_dataset\all.txt'
    jpg_dir = r'X:\Transportation_infrastructure_defect_dataset\jpg_dat'
    RDD_dat(xml_dir, img_dir, output_dir, index_path, jpg_dir)

    # list_dir = os.listdir(r'X:\Transportation_infrastructure_defect_dataset\RDD_test')
    # with open(r'X:\Transportation_infrastructure_defect_dataset\test_vol.txt', 'w', encoding='utf-8') as f:
    #     for file in list_dir:
    #         f.write(file + '\n')

    # input_file_path = r'X:\Transportation_infrastructure_defect_dataset\all.txt'
    # output_file_path = r'X:\Transportation_infrastructure_defect_dataset\val.txt'
    # random_delete_90_percent(input_file_path, output_file_path)