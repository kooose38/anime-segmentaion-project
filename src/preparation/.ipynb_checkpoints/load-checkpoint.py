from glob import glob 
import torch 
import os 
import numpy as np 
from typing import Dict, List, Any, Tuple
import requests
from PIL import Image 
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader 


def get_data(root_path: str="./data/raw/npz") -> Tuple[List[str], List[np.ndarray]]:
    '''アノテーションデータの読み込み処理関数'''
    url_list, anno_list = [], []
    for f in glob(root_path+"/"+"*.npz"):
        np_data = np.load(f)
        url = np_data["URL"].item()
        skin = np_data["skin"]
        hair = np_data["hair"] * 2
        face = np_data["face"] * 3
        anno = skin+hair+face 
        url_list.append(url)
        anno_list.append(anno)
    return url_list, anno_list 


def req(url_list: List[str]):
    '''画像データのリクエストをする関数'''
    for url in url_list:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "lxml")
        link = soup.findAll("img")
        for i in link:
            src = i.get("src")
            if "https://" in src:
                r = requests.get(src)
                os.makedirs("./data/images", exist_ok=True)
                with open(f"./data/images/anime_{url.split('/')[-1]}_.jpg", "wb") as f:
                    f.write(r.content)


def create_data(url_list: List[str], anno_list: List[np.ndarray]) -> Tuple[List[Any], List[Any]]:
    '''データをPIL形式のリストに格納する'''
    img_list, ann_list = [], []
    for i, (url, anno) in enumerate(zip(url_list, anno_list)):
        for img_path in glob("./data/raw/images/"+f"anime_{url.split('/')[-1]}_.jpg"):
            img = Image.open(img_path)
            ann_img = Image.fromarray(anno, mode="P")
        img_list.append(img)
        ann_list.append(ann_img)
    return img_list, ann_list


