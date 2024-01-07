import sys
from argparse import ArgumentParser
import os
sys.path.insert(0,os.getcwd())
import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openslide as opsl
from PIL import Image
from collections import Counter
import shutil
import spams
from sklearn.manifold import TSNE
from utils.inference import inference_model, init_model
from core.visualization.image import imshow_infos
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from vahadane import vahadane

ROI_classes_names = ['not_ROI','ROI']
MGMT_classes_names = ['me','unme']

ROI_model_path = ''
ROI_model_cfg,ROI_train_pipeline,ROI_val_pipeline,ROI_data_cfg,ROI_lr_config,ROI_optimizer_cfg = file2dict(ROI_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROI_model = BuildNet(ROI_model_cfg)
ROI_model = init_model(ROI_model, ROI_data_cfg, device=device, mode='eval')

MGMT_model_path = ''
MGMT_model_cfg,MGMT_train_pipeline,MGMT_val_pipeline,MGMT_data_cfg,MGMT_lr_config,MGMT_optimizer_cfg = file2dict(MGMT_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MGMT_model = BuildNet(MGMT_model_cfg)
MGMT_model = init_model(MGMT_model, MGMT_data_cfg, device=device, mode='eval')

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img

TARGET_PATH = ''
target_image = read_image(TARGET_PATH)
vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
Wt, Ht = vhd.stain_separate(target_image)

def total_result(data):
    data.loc[data['pred_class'] == 'unme', 'pred_score'] = 1 - data.loc[data['pred_class'] == 'unme', 'pred_score']
    deviation_sum = abs(data['pred_score'] - 0.5).sum()
    data['weight'] = (data['pred_score'] - 0.5).abs() / deviation_sum
    weighted_average = (data['pred_score'] * data['weight']).sum()
    return weighted_average

def cutPatchAndPredictPatch1(svs_file, patch_c, patch_r, step, ROI_model, MGMT_model, ROI_val_pipeline, MGMT_val_pipeline, ROI_classes_names, MGMT_classes_names):
    columns = ['pred_label', 'pred_score', 'pred_class']
    MGMT_result_dataframe = pd.DataFrame(columns=columns)
    slide = opsl.OpenSlide(svs_file)
    w_count = int(slide.level_dimensions[0][0] // step)
    h_count = int(slide.level_dimensions[0][1] // step)
    i = 0
    slide_image = np.zeros((slide.level_dimensions[0][1], slide.level_dimensions[0][0], 3), dtype=np.uint8)
    patch_arr = np.zeros((patch_r, patch_c, 3), dtype=np.uint8)
    for x in range(1, w_count - 1):
        for y in range(int(h_count)):
            slide.read_region((x * step, y * step), 0, (patch_c, patch_r)).convert('RGB').load()
            slide_region = np.array(slide.read_region((x * step, y * step), 0, (patch_c, patch_r)))[:, :, :3][:, :, ::-1]
            i += 1
            if Counter(np.array(slide_region).flatten()).most_common(3)[0][0] <= 200:
                source_image = cv2.cvtColor(slide_region, cv2.COLOR_BGR2RGB)
                p = np.percentile(source_image, 90)
                source_image = np.clip(source_image * 255.0 / p, 0, 255).astype(np.uint8)
                Ws, Hs = vhd.stain_separate(source_image)
                vhd.fast_mode=0;vhd.getH_mode=0;
                img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
                slide_region = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ROI_result = inference_model(ROI_model, slide_region, ROI_val_pipeline, ROI_classes_names)
                if ROI_result['pred_label'] == 1:
                    MGMT_reslut = inference_model(MGMT_model, slide_region, MGMT_val_pipeline, MGMT_classes_names)
                    result_pd = pd.DataFrame(MGMT_reslut, index=[0])
                    MGMT_result_dataframe = pd.concat([MGMT_result_dataframe,result_pd],ignore_index=True)
            patch_arr[:, :, :] = 0
    slide.close()
    last_result = total_result(MGMT_result_dataframe)
    return last_result

def main():
    svs_file = ''
    MGMT_result = cutPatchAndPredictPatch1(svs_file,512,512,512,ROI_model, MGMT_model,ROI_val_pipeline, MGMT_val_pipeline, ROI_classes_names, MGMT_classes_names)
    print(MGMT_result)
    return MGMT_result

if __name__ == '__main__':
    main()