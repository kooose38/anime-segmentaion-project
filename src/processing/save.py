import os 
import pickle 

def save_pkl(dataset, filename: str):
    '''DataLoaderの保存をする関数'''
    os.makedirs("./data/processed", exist_ok=True)
    with open(f"./data/processed/{filename}.pkl", "wb") as f:
        pickle.dump(dataset, f)