from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm

save_dir = "/home/jianglongyu/mydrive/clip_dataset/train_dataset"

def train_data():
    dataset = load_dataset("JaronU/CLIP_train_dataset", split="train")
    os.makedirs(save_dir, exist_ok=True)
    label2name = {
        0: "doughnut",
        1: "glass_cup",
        2: "lemon",
        3: "chinese_noodle",
        4: "chinese_flute"
    }

    def save_train_data(dataset_split, split_name):
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, sample in enumerate(tqdm(dataset_split)):
            image = sample["image"]
            label = sample["label"]
            class_name = label2name[label]

            class_dir = os.path.join(split_dir, f"{label:03d}_{class_name}")
            os.makedirs(class_dir, exist_ok=True)
            image_path = os.path.join(class_dir, f"{i + 1:03d}.png")
            image.save(image_path)

    save_train_data(dataset, "train")

def test_data():
    dataset = load_dataset("JaronU/CLIP_inference_test_dataset", split="train")
    os.makedirs(save_dir, exist_ok=True)
    label2name = {
        0: "doughnut",
        1: "glass_cup",
        2: "lemon",
        3: "chinese_noodle",
        4: "chinese_flute",
        5: "others"
    }

    def save_test_data(dataset_split, split_name):
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, sample in enumerate(tqdm(dataset_split)):
            image = sample["image"]
            label = sample["label"] if sample["label"] <= 4 else 5
            class_name = label2name[label]

            class_dir = os.path.join(split_dir, f"{label:03d}_{class_name}")
            os.makedirs(class_dir, exist_ok=True)
            image_path = os.path.join(class_dir, f"{i + 1:03d}.png")
            image.save(image_path)

    save_test_data(dataset, "test")
test_data()
print("done")