import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageEnhance
from tqdm import tqdm
import numpy as np
import base64
import zlib
from io import BytesIO

# ====================== 【配置项】 ======================
BASE_PATH = r'D:\myProjects\大创(1)\pythonProject\pythonProject\SUIM_Processed'
OUTPUT_PATH = r'D:\myProjects\大创(1)\pythonProject\pythonProject\output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 8
EPOCHS = 60
BATCH_SIZE = 8
IMAGE_SIZE = 256
ENCODER_LR = 1e-4       # 预训练编码器用较低学习率
DECODER_LR = 1e-3        # 解码器用较高学习率
VAL_RATIO = 0.15

# ImageNet 归一化参数（配合预训练权重）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SUB_DIRS = ["1_raw", "2_enhanced", "3_noise", "4_scattering", "5_blur"]

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ====================== Focal Loss（解决类别不平衡） ======================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ====================== U-Net 解码器模块 ======================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ====================== ResNet34 + U-Net（预训练编码器） ======================
class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Encoder (来自预训练 ResNet34)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch, H/2
        self.pool0 = resnet.maxpool                                        # 64ch, H/4
        self.enc1 = resnet.layer1   # 64ch,  H/4
        self.enc2 = resnet.layer2   # 128ch, H/8
        self.enc3 = resnet.layer3   # 256ch, H/16
        self.enc4 = resnet.layer4   # 512ch, H/32

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)   # H/16
        self.dec3 = DecoderBlock(256, 128, 128)   # H/8
        self.dec2 = DecoderBlock(128, 64, 64)     # H/4
        self.dec1 = DecoderBlock(64, 64, 64)      # H/2
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # H
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(32, num_classes, 1)

    def encoder_params(self):
        """返回编码器参数（低学习率）"""
        return list(self.enc0.parameters()) + list(self.enc1.parameters()) + \
               list(self.enc2.parameters()) + list(self.enc3.parameters()) + \
               list(self.enc4.parameters())

    def decoder_params(self):
        """返回解码器参数（高学习率）"""
        return list(self.dec4.parameters()) + list(self.dec3.parameters()) + \
               list(self.dec2.parameters()) + list(self.dec1.parameters()) + \
               list(self.dec0.parameters()) + list(self.head.parameters())

    def forward(self, x):
        e0 = self.enc0(x)           # 64ch,  H/2
        e1 = self.enc1(self.pool0(e0))  # 64ch,  H/4
        e2 = self.enc2(e1)          # 128ch, H/8
        e3 = self.enc3(e2)          # 256ch, H/16
        e4 = self.enc4(e3)          # 512ch, H/32

        d4 = self.dec4(e4, e3)      # 256ch, H/16
        d3 = self.dec3(d4, e2)      # 128ch, H/8
        d2 = self.dec2(d3, e1)      # 64ch,  H/4
        d1 = self.dec1(d2, e0)      # 64ch,  H/2
        d0 = self.dec0(d1)          # 32ch,  H

        return self.head(d0)


# ====================== 类别映射 ======================
CLASS_NAME_TO_ID = {
    "aquatic_plants_and sea-grass": 0,
    "fish_and_vertebrates": 1,
    "human_divers": 2,
    "reefs_and_invertebrates": 3,
    "robots": 4,
    "sea-floor_and_rocks": 5,
    "waterbody_background": 6,
    "wrecks_and_ruins": 7
}


# ====================== 数据增强（图片+掩码同步） ======================
def augment_pair(img_pil, mask_np):
    """对图片和掩码做同步的随机增强"""
    # 随机水平翻转
    if random.random() > 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_np = np.fliplr(mask_np).copy()

    # 随机垂直翻转
    if random.random() > 0.5:
        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        mask_np = np.flipud(mask_np).copy()

    # 随机旋转（0°/90°/180°/270°）
    k = random.randint(0, 3)
    if k > 0:
        img_pil = img_pil.rotate(k * 90, expand=False)
        mask_np = np.rot90(mask_np, k).copy()

    # 颜色增强（仅图片）
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        img_pil = ImageEnhance.Color(img_pil).enhance(factor)

    return img_pil, mask_np


# ====================== 标签加载 ======================
def load_label_from_json(label_dir, img_filename):
    """从 JSON 文件加载真实标签掩码"""
    json_path = os.path.join(label_dir, img_filename + ".json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_w = data['size']['width']
        img_h = data['size']['height']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for obj in data.get('objects', []):
            class_title = obj.get('classTitle')
            if class_title not in CLASS_NAME_TO_ID:
                continue

            class_id = CLASS_NAME_TO_ID[class_title]
            bitmap = obj.get('bitmap', {})
            origin = bitmap.get('origin', [0, 0])
            bitmap_data = bitmap.get('data', '')
            if not bitmap_data:
                continue

            try:
                decoded = base64.b64decode(bitmap_data)
                decompressed = zlib.decompress(decoded)
                bitmap_img = Image.open(BytesIO(decompressed))
                bitmap_array = np.array(bitmap_img)

                if len(bitmap_array.shape) == 3 and bitmap_array.shape[2] == 4:
                    bitmap_mask = (bitmap_array[:, :, 3] > 0)
                elif len(bitmap_array.shape) == 3:
                    bitmap_mask = (bitmap_array.sum(axis=2) > 0)
                else:
                    bitmap_mask = (bitmap_array > 0)

                bh, bw = bitmap_mask.shape
                y0, x0 = origin[1], origin[0]
                y1 = min(y0 + bh, img_h)
                x1 = min(x0 + bw, img_w)

                if y1 > y0 and x1 > x0:
                    mask[y0:y1, x0:x1][bitmap_mask[:y1 - y0, :x1 - x0]] = class_id

            except Exception:
                continue

        return mask

    except Exception:
        return None


# ====================== 数据集 ======================
class SegDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, label_dir, augment=False):
        self.image_list = image_list
        self.label_dir = label_dir
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img_filename = os.path.basename(img_path)

        # 读取图片
        try:
            img_pil = Image.open(img_path).convert('RGB').resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        except Exception:
            img_pil = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

        # 读取掩码
        mask_full = load_label_from_json(self.label_dir, img_filename)
        if mask_full is not None:
            mask_pil = Image.fromarray(mask_full, mode='L')
            mask_resized = np.array(mask_pil.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.int64)
        else:
            mask_resized = np.full((IMAGE_SIZE, IMAGE_SIZE), 6, dtype=np.int64)

        # 数据增强
        if self.augment:
            img_pil, mask_resized = augment_pair(img_pil, mask_resized)

        # 转 tensor + ImageNet 归一化
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized.copy()).long()

        return img_tensor, mask_tensor, img_path


# ====================== 按文件名划分 train/val（防止数据泄露）======================
def build_splits(base_path, label_dir, val_ratio=0.15):
    """
    以 base filename 为单位划分：
    - 训练集：使用所有 5 个文件夹的图片
    - 验证集：仅使用 1_raw（原始干净图）
    """
    # 收集 1_raw 中有标签的文件名
    raw_dir = os.path.join(base_path, "1_raw")
    all_names = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif'))
        and os.path.exists(os.path.join(label_dir, f + ".json"))
    ])

    random.seed(42)
    random.shuffle(all_names)
    val_count = int(len(all_names) * val_ratio)
    val_names = set(all_names[:val_count])
    train_names = set(all_names[val_count:])

    # 训练集：所有文件夹中属于 train_names 的图
    train_list = []
    for folder in SUB_DIRS:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname in train_names:
                train_list.append(os.path.join(folder_path, fname))

    # 验证集：仅 1_raw 中属于 val_names 的图
    val_list = [os.path.join(raw_dir, f) for f in val_names]

    return train_list, val_list


# ====================== 计算 mIoU ======================
def compute_miou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


# ====================== 验证 ======================
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labs, _ in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)
            outputs = model(imgs)
            total_loss += criterion(outputs, labs).item()
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    miou = compute_miou(all_preds, all_labels, NUM_CLASSES)
    return total_loss / len(val_loader), miou


# ====================== 主训练函数 ======================
def train_and_analyze():
    label_dir = os.path.join(BASE_PATH, "6_label", "ann")

    # 按文件名划分，防止数据泄露
    train_list, val_list = build_splits(BASE_PATH, label_dir, VAL_RATIO)
    print(f"\n✅ 训练集: {len(train_list)} 张（5个文件夹 × {len(train_list)//5} 场景）")
    print(f"✅ 验证集: {len(val_list)} 张（仅 1_raw 干净图）")
    print(f"✅ 标签目录: {label_dir}")

    train_ds = SegDataset(train_list, label_dir, augment=True)
    val_ds = SegDataset(val_list, label_dir, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available())

    # 模型
    model = ResNet34UNet(NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型: ResNet34-UNet, 参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"🖥️  设备: {DEVICE}")

    # 差分学习率：编码器低LR，解码器高LR
    optimizer = torch.optim.AdamW([
        {'params': model.encoder_params(), 'lr': ENCODER_LR},
        {'params': model.decoder_params(), 'lr': DECODER_LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0)

    # 路径
    best_model_path = os.path.join(OUTPUT_PATH, "best_model.pth")
    final_model_path = os.path.join(OUTPUT_PATH, "final_model.pth")
    log_path = os.path.join(OUTPUT_PATH, "训练日志.json")

    best_miou = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 12  # 早停耐心值
    log_records = []

    print(f"\n🚀 开始训练 (共 {EPOCHS} epochs, 早停耐心={PATIENCE})...\n")

    for epoch in range(EPOCHS):
        # === 训练 ===
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for step, (imgs, labs, _) in enumerate(pbar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)

            loss = criterion(model(imgs), labs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}",
                             Avg=f"{train_loss/(step+1):.4f}")

        scheduler.step()
        avg_train = train_loss / len(train_loader)

        # === 验证 ===
        val_loss, val_miou = validate(model, val_loader, criterion)
        print(f"  Epoch {epoch+1}: Train={avg_train:.4f} | Val={val_loss:.4f} | mIoU={val_miou:.4f}")

        log_records.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train, 4),
            "val_loss": round(val_loss, 4),
            "val_miou": round(val_miou, 4),
        })

        # 保存最佳
        if val_miou > best_miou:
            best_miou = val_miou
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 新最佳模型 mIoU={best_miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  早停触发 (连续 {PATIENCE} epoch 无提升)")
                break

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_records, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print(f"📄 训练日志: {log_path}")
    print(f"📦 最终模型: {final_model_path}")
    print(f"📦 最佳模型: {best_model_path} (Val mIoU={best_miou:.4f})")
    print(f"📊 最佳验证损失: {best_val_loss:.4f}")
    print("=" * 70)
    return avg_train


# ====================== 主程序入口 ======================
if __name__ == "__main__":
    print("=" * 70)
    print("  语义分割模型训练脚本 - 多文件夹图片分析")
    print("=" * 70)

    try:
        avg_loss = train_and_analyze()
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        avg_loss = None

    input("\n按Enter键退出...")  # 防止终端闪退
