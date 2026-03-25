import json
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

def cocojson_to_mask(cocojson_path,image_dir,output_mask_dir):
    os.makedirs(output_mask_dir,exist_ok = True)
    coco = COCO(cocojson_path)
    #获取分类数
    cat_ids = coco.getCatIds()
    for img_id in tqdm(coco.getImgIds()):
        #获取图片信息，对应标注的信息和分类信息
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds = img_id)
        anns = coco.loadAnns(ann_ids)
        #创建掩码图片
        mask = np.zeros((img_info['height'],img_info['width']),dtype = np.uint8)
        for ann in anns:
            #获取标注类别
            class_id = ann['category_id']
            if 'segmentation' in ann:
                #提取轮廓坐标
                poly = ann['segmentation'][0] if isinstance(ann['segmentation'],list) else ann['segmentation']
                pts = np.array(poly).reshape(-1,2).astype(np.int32)
                #使用
                cv2.fillPoly(mask,[pts],color = class_id)
        
        mask_path = os.path.join(output_mask_dir,img_info['file_name'].replace('.jpg','.png'))
        cv2.imwrite(mask_path,mask)

if __name__ == "__main__":
    cocojson_to_mask(
        cocojson_path = '/data/unet-attention-dsconv_github/data/valid/_annotations.coco.json',
        image_dir = '/data/unet-attention-dsconv_github/data/valid/',
        output_mask_dir = '/data/unet-attention-dsconv_github/data/valid_mask/'
    )
    print('over!!!!!!!!!!!!!')