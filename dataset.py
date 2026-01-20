import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor
import torchvision.transforms as transforms

class CLIPDataset(Dataset):
    def __init__(self, txt_file, data_dir, model_name='openai/clip-vit-base-patch32', max_len=64, is_train=True):
        self.data_info = pd.read_csv(txt_file)
        self.data_dir = data_dir
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.max_len = max_len
        self.is_train = is_train
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'null': -1}

        # 改进 3：图像增强 (Data Augmentation)
        if self.is_train:
            self.image_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # 随机颜色抖动
                transforms.RandomRotation(15), # 随机旋转
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        guid = str(self.data_info.iloc[idx, 0])
        label_str = str(self.data_info.iloc[idx, 1])

        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")

        # 路径保护与异常处理
        if not os.path.exists(text_path) or not os.path.exists(img_path):
            text = "none"
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            with open(text_path, 'r', encoding='latin-1') as f:
                text = f.read().strip()
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 应用图像增强
        pixel_values = self.image_transform(image)

        # 文本分词
        text_inputs = self.processor(
            text=[text if text else "none"],
            return_tensors="pt",
            padding='max_length',
            max_length=self.max_len,
            truncation=True
        )

        return {
            'guid': guid,
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            'pixel_values': pixel_values,
            'label': torch.tensor(self.label_map[label_str], dtype=torch.long)
        }