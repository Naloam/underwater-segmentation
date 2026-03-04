# UV 虚拟环境安装指南

由于依赖库较大（PyTorch ~1.2GB，transformers ~50MB + CLIP权重 ~350MB），推荐使用UV创建轻量级虚拟环境。

## 什么是UV？

UV是Rust编写的极速Python包管理器，比pip快10-100倍。

## 安装UV

```powershell
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用pip安装
pip install uv
```

## 创建虚拟环境

```bash
cd d:\myProjects\大创(1)\Part2_Enhanced

# 创建项目虚拟环境（自动使用.python-version）
uv venv

# 激活环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

## 安装依赖（分阶段）

### 阶段1: 基础依赖（训练必需）

```bash
# 同步依赖（使用uv lock机制）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install pillow numpy tqdm scikit-image matplotlib seaborn pyyaml
```

### 阶段2: 可选依赖（CLIP语义分支）

```bash
# 如果需要启用CLIP分支
uv pip install transformers einops

# 首次运行时会自动下载CLIP权重 (~350MB)
```

## 快速验证

```bash
# 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "from models import create_model; print('Model import OK')"

# 测试模型
python -c "
from models import create_model
from configs.model_config import ModelConfig
import torch

config = ModelConfig(use_clip=False, use_diffusion=False)
model = create_model(config)
x = torch.randn(2, 3, 256, 256)
logits = model(x)
print(f'✓ Model test passed: {logits.shape}')
"
```

## 开始训练

```bash
# 基础训练（无需CLIP）
python train.py --batch-size 4 --epochs 50
```

## UV优势

| 特性 | UV | pip/venv |
|------|-----|----------|
| 安装速度 | ⚡ 10-100x | 🐌 较慢 |
| 依赖解析 | ✅ 精确 | ⚠️ 可能冲突 |
| 锁文件 | ✅ 自动生成 | ❌ 需手动 |
| 环境隔离 | ✅ 完善 | ✅ 完善 |

## 常见问题

### Q: UV和现有conda环境冲突？
A: 建议使用独立的uv venv，避免与conda混用。

### Q: 显存不足？
A: 设置 `use_clip=False, use_diffusion=False` 减少模型大小。

### Q: transformers安装失败？
A: 暂时跳过，先用baseline模型训练。

## 环境清理（如需重置）

```bash
# 删除虚拟环境
deactivate  # 先退出
rmdir /s .venv

# 重新创建
uv venv
.venv\Scripts\activate
```

---

*推荐使用UV管理依赖，安装更快更稳定！*
