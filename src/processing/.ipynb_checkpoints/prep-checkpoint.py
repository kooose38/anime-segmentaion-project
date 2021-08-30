from src.processing.data_augumentation import Scale, RandomMirror, RandomRotation, Resize, Normalize_Tensor, Compose
import torch 
from torch.utils.data import DataLoader 
from typing import Tuple, List, Any
import numpy as np 

class Transform():
    def __init__(self, resize: int=475, mean: Tuple[float]=(0.485, 0.456, 0.406), std: Tuple[float]=(0.229, 0.456, 0.406)):
        '''画像をテンソル型にする'''
        self.data_transform = {
            "train": Compose([
                              Scale(scale=[0.5, 1.5]),
                              RandomRotation(angle=[-10, 10]),
                              RandomMirror(),
                              Resize(resize),
                              Normalize_Tensor(mean, std)
            ]),
            "test": Compose([
                             Resize(resize),
                             Normalize_Tensor(mean, std)
            ])
        }
    def __call__(self, phase: str, img: Any, annot: Any) -> torch.Tensor:
        return self.data_transform[phase](img, annot)

transform = Transform()


def train_val_split(img_list: List[Any], anno_list: List[Any], num: int=2):
    '''DataLoaderを作成して、訓練検証の分割をする'''
    dataset = []
    for e in range(num):
        for img, ann in zip(img_list, anno_list):
            if np.array(img).shape[2] > 3: continue  # チャネルが４の画像を除去する
            img_tensor, ann_tensor = transform("train", img, ann)
            data = {}
            data["input"] = img_tensor 
            data["label"] = ann_tensor 
            dataset.append(data)
    
    n_ = len(dataset)
    n_train = int(n_*.7)
    train_ds, val_ds = dataset[:n_train], dataset[n_train:]
    print(f"train: {len(train_ds)} val: {len(val_ds)}")
    train = DataLoader(train_ds, batch_size=8, shuffle=True)
    val = DataLoader(val_ds, batch_size=8, shuffle=True)
    print("---")
    for r in train:
        print(r["input"].size())
        print(r["label"].size())
        print(r["input"][0])
        break 
    return train, val 