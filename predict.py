import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from models import MultiModalModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_TXT = 'test_without_label.txt'
DATA_DIR = 'data/'
MODEL_PATH = 'best_model.pth'

def predict():
    # 1. 准备测试数据
    test_ds = MultiModalDataset(TEST_TXT, DATA_DIR, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # 2. 加载模型
    model = MultiModalModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 3. 推理
    results = []
    id_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    with torch.no_grad():
        for batch in test_loader:
            guids = batch['guid']
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            imgs = batch['image'].to(DEVICE)
            
            outputs = model(ids, mask, imgs, mode='both')
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(guids)):
                results.append({'guid': guids[i], 'tag': id_map[predicted[i].item()]})

    # 4. 写入文件
    df = pd.DataFrame(results)
    df.to_csv('test_with_predictions.txt', index=False)
    print("Inference finished! Results saved to test_with_predictions.txt")

if __name__ == "__main__":
    predict()