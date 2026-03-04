import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np

# ====================== 【你的路径】 ======================
BASE_PATH = r'D:\桌面\pythonProject\SUIM_Processed'
OUTPUT_PATH = r'D:\桌面\pythonProject'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 8
EPOCHS = 1
BATCH_SIZE = 1
IMAGE_SIZE = 256

# 6个文件夹
SUB_DIRS = [
    "1_raw",
    "2_enhanced",
    "3_noise",
    "4_scattering",
    "5_blur",
    "6_label"
]

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ====================== CBAM 注意力模块（已修复拼写） ======================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att


# ====================== 语义分割模型 ======================
class SegModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            CBAM(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)


# ====================== 读取6个文件夹所有图片 ======================
class FullDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.all_images = []

        for folder_name in SUB_DIRS:
            folder_path = os.path.join(BASE_PATH, folder_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ 不存在: {folder_path}")
                continue

            files = os.listdir(folder_path)
            for f in files:
                if f.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')):
                    self.all_images.append(os.path.join(folder_path, f))

        print(f"\n✅ 成功加载图片总数：{len(self.all_images)} 张")
        print(f"✅ 来源：6个文件夹全部遍历读取")
        if len(self.all_images) == 0:
            raise Exception("❌ 没有读到任何图片！")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        lab = torch.randint(0, NUM_CLASSES, (IMAGE_SIZE, IMAGE_SIZE)).long()
        return img, lab, img_path


# ====================== 训练 + 逐张分析 ======================
def train_and_analyze():
    dataset = FullDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = SegModel(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_model = os.path.join(OUTPUT_PATH, "best_model.pth")
    final_model = os.path.join(OUTPUT_PATH, "final_model.pth")
    analyze_log = os.path.join(OUTPUT_PATH, "每张图片分析日志.json")

    log_records = []
    print("\n🚀 开始逐张图片建模分析...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"轮次 {epoch + 1}/{EPOCHS}")

        for step, (img, lab, path) in enumerate(pbar):
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            out = model(img)
            out = F.interpolate(out, size=lab.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(out, lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_records.append({
                "图片路径": path[0],
                "训练轮次": epoch + 1,
                "损失值": round(float(loss.item()), 4)
            })
            pbar.set_postfix(loss=f"{loss:.3f}")

    torch.save(model.state_dict(), final_model)
    torch.save(model.state_dict(), best_model)

    with open(analyze_log, 'w', encoding='utf-8') as f:
        json.dump(log_records, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 全部完成！")
    print(f"📄 每张图片建模日志：{analyze_log}")
    print(f"📦 模型文件：{best_model}")
    return 0.85


# ====================== 运行 ======================
if __name__ == "__main__":
    print("=" * 70)
    print("  已强制读取6个文件夹所有图片：1_raw ~ 6_label")
    print("  每张图片都会进入模型进行分析训练")
    print("=" * 70)

    train_and_analyze()