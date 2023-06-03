import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2


class DaconDataset(Dataset):
    def __init__(self, base_folder, label_df, max_len=320, align_csv=False, phase='train', transforms=None):
        assert phase in ['train', 'valid', 'test']

        self.base_folder = base_folder
        self.label_df = label_df
        self.transforms = transforms
        self.phase = phase
        self.max_len = max_len
        self.align_csv = align_csv

        self.label_description = {
                "1_00_0" : "딸기", 
                "2_00_0" : "토마토",
                "2_a5_2" : "토마토_흰가루병_중기",
                "3_00_0" : "파프리카",
                "3_a9_1" : "파프리카_흰가루병_초기",
                "3_a9_2" : "파프리카_흰가루병_중기",
                "3_a9_3" : "파프리카_흰가루병_말기",
                "3_b3_1" : "파프리카_칼슘결핍_초기",
                "3_b6_1" : "파프리카_다량원소결필(N)_초기",
                "3_b7_1" : "파프리카_다량원소결필(P)_초기",
                "3_b8_1" : "파프리카_다량원소결필(K)_초기",
                "4_00_0" : "오이",
                "5_00_0" : "고추",
                "5_a7_2" : "고추_탄저병_중기",
                "5_b6_1" : "고추_다량원소결필(N)_초기",
                "5_b7_1" : "고추_다량원소결필(P)_초기",
                "5_b8_1" : "고추_다량원소결필(K)_초기",
                "6_00_0" : "시설포도",
                "6_a11_1" : "시설포도_탄저병_초기",
                "6_a11_2" : "시설포도_탄저병_중기",
                "6_a12_1" : "시설포도_노균병_초기",
                "6_a12_2" : "시설포도_노균병_중기",
                "6_b4_1" : "시설포도_일소피해_초기",
                "6_b4_3" : "시설포도_일소피해_말기",
                "6_b5_1" : "시설포도_축과병_초기"
            }

        self.label_one_hot_encoder = {key:idx for idx, key in enumerate(self.label_description)}
        self.label_one_hot_decoder = {val:key for key, val in self.label_one_hot_encoder.items()}
    
        self.csv_feature_dict = {'내부 온도 1 평균': [3.4, 47.3],
                                 '내부 온도 1 최고': [3.4, 47.6],
                                 '내부 온도 1 최저': [3.3, 47.0],
                                 '내부 습도 1 평균': [23.7, 100.0],
                                 '내부 습도 1 최고': [25.9, 100.0],
                                 '내부 습도 1 최저': [0.0, 100.0],
                                 '내부 이슬점 평균': [0.1, 34.5],
                                 '내부 이슬점 최고': [0.2, 34.7],
                                 '내부 이슬점 최저': [0.0, 34.4]}

    def __len__(self):
        return len(self.label_df)

    def __align_csv__(self, df):
        df = np.array(df)
        df = cv2.resize(df, dsize=(9, self.max_len), interpolation=cv2.INTER_LINEAR)

        return df

    def __getitem__(self, idx):
        pre_fix = str(self.label_df.iloc[idx]['image'])
        
        image_fn = os.path.join(self.base_folder, pre_fix, pre_fix + '.jpg')
        json_fn = os.path.join(self.base_folder, pre_fix, pre_fix + '.json')
        csv_fn = os.path.join(self.base_folder, pre_fix, pre_fix + '.csv')

        ## read image
        assert os.path.isfile(image_fn)
        sample_image = cv2.imread(image_fn)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        ## read csv
        sample_csv = pd.read_csv(csv_fn)[self.csv_feature_dict.keys()]
        sample_csv = sample_csv.replace('-', 0)

        ## min-max scailing for csv
        for col in sample_csv.columns:
            sample_csv[col] = sample_csv[col].astype(float) - self.csv_feature_dict[col][0]
            sample_csv[col] = sample_csv[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])

        if self.align_csv:
            sample_csv = self.__align_csv__(sample_csv)

        else:
            pad = np.zeros((self.max_len, len(sample_csv.columns)))
            length = min(self.max_len, len(sample_csv))
            pad[-length:] = sample_csv.to_numpy()[-length:]
            sample_csv = pad

        sample = dict()

        # if self.phase == 'train':
        #     sample_json = json.load(open(json_fn, 'r'))

        #     if len(sample_json['annotations']['bbox']) == 1:
        #         bbox = sample_json['annotations']['bbox'][0]
        #         for key, value in bbox.items(): bbox[key] = int(value)
        #         sample_image = sample_image[bbox['y']:bbox['y']+bbox['h'], bbox['x']:bbox['x']+bbox['w']]

        if self.phase in ['train', 'valid']:
            sample_label = self.label_df.iloc[idx]['label']
            one_hot_label = self.encode(sample_label)
            sample['label'] = one_hot_label
            
        if self.transforms is not None:
            sample_image = self.transforms(image=sample_image)['image']
            sample_csv = torch.tensor(sample_csv, dtype=torch.float32)

        sample['image'] = sample_image
        sample['csv'] = sample_csv

        return sample

    def encode(self, label):
        return self.label_one_hot_encoder[label]

    def decode(self, one_hot_value):
        return self.label_one_hot_decoder[one_hot_value]

    def description(self, label):
        return self.label_description[label]


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# sz = 512
sz = 256

def get_train_transforms():
    return A.Compose(
        [
            A.Resize(height=sz, width=sz),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(15),
                translate_percent=(0.05, 0.05),
                cval=0,
                cval_mask=0,
                p=0.3,
            ),
            A.Flip(p=0.5),
            A.CoarseDropout(p=0.5, max_holes=16, max_height=50, max_width=50, min_holes=4, min_height=25, min_width=25),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=sz, width=sz),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    )

def get_test_transforms():
    return A.Compose(
        [
            A.Resize(height=sz, width=sz),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    )