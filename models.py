import torch
import torch.nn as nn
from transformers import CLIPModel

class GatedMultimodalClassifier(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', num_classes=3):
        super(GatedMultimodalClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        
        # 冻结 CLIP 参数（可选，显存不足时开启）
        for param in self.clip.parameters():
            param.requires_grad = False

        # 改进 1：门控融合层 (Gated Fusion)
        # 输入是文本(512)和图像(512)的拼接，输出一个 0-1 的权重
        self.gate = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # 融合后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), # 门控加权后维度为 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values, mode='both'):
        # 1. 提取特征
        if mode == 'img_only':
            text_embeds = torch.zeros(input_ids.size(0), 512).to(input_ids.device)
            image_outputs = self.clip.get_image_features(pixel_values=pixel_values)
            image_embeds = image_outputs
        elif mode == 'text_only':
            text_outputs = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs
            image_embeds = torch.zeros(pixel_values.size(0), 512).to(pixel_values.device)
        else:
            text_embeds = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            image_embeds = self.clip.get_image_features(pixel_values=pixel_values)

        # 2. 门控融合逻辑
        # 只有在 both 模式下门控才有意义
        if mode == 'both':
            concat_features = torch.cat((text_embeds, image_embeds), dim=1)
            g = self.gate(concat_features) # 计算文本的权重
            fused_features = g * text_embeds + (1 - g) * image_embeds
        else:
            # 单模态模式下直接相加（因为另一侧是0）
            fused_features = text_embeds + image_embeds

        # 3. 分类输出
        logits = self.classifier(fused_features)
        return logits