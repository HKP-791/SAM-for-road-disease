import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import zoom
import os
import glob

dat_dir = r'X:\SAMed_fortest\out_put\predictions'
img_patten = glob.glob(os.path.join(dat_dir, '*_img.nii.gz'))
mask_patten = glob.glob(os.path.join(dat_dir, '*_pred.nii.gz'))
for img_file in img_patten:
    mask_file = img_file.split('_img.nii.gz')[0] + '_pred.nii.gz'

    mask = nib.load(mask_file).get_fdata()
    mask = mask*255/np.max(mask)
    img_arry = nib.load(img_file).get_fdata()
    img_arry = np.transpose(img_arry*255/np.max(img_arry), (1, 2, 0)).astype(np.uint8)
    img_arry = zoom(img_arry, (mask.shape[0] / img_arry.shape[0], mask.shape[1] / img_arry.shape[1], 1), order=3)
    img = Image.fromarray(img_arry).convert('RGBA')

    # 创建一个与原始图片相同尺寸的透明掩膜
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # 定义掩膜的颜色和透明度
    mask_color = (255, 0, 0, 128)  # 红色，透明度为128

    # 将掩膜应用到透明图层
    draw = ImageDraw.Draw(overlay)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0:  # 假设掩膜中的非零值表示分割区域
                draw.point((x, y), fill=(255, img_arry[y, x, 1], img_arry[y, x, 2], 128))
            else:
                draw.point((x, y), fill=(img_arry[y, x, 0], img_arry[y, x, 1], img_arry[y, x, 2], 255))

    # 将透明掩膜叠加到原始图片上
    combined_image = Image.alpha_composite(img, overlay)

    # 保存或显示结果图片
    result_name = img_file.split('X:\\SAMed_fortest\\out_put\\predictions\\')[-1].split('_img.nii.gz')[0]
    combined_image.save(os.path.join('X:\\SAMed_fortest\\segmentation_result\\' + result_name + '.png'))