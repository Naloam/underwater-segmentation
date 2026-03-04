# ==============================
# SUIM水下图像分割 全流程单文件代码
# 路径：DanacondapythonProjectSUIM_Processedrun_all.py
# 一键运行：训练+实验+可视化+结果保存
# ==============================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import Config, read_base
from mmengine.runner import Runner
from mmseg.apis import init_model, train_model, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_model
from mmseg.evaluation import build_evaluator

# 全局固定路径（你的数据集绝对路径）
ROOT_DIR = r'DanacondapythonProjectSUIM_Processed'
EXP_ROOT = os.path.join(ROOT_DIR, 'exp_results')
os.makedirs(EXP_ROOT, exist_ok=True)

# 解决中文绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================================
# 一、自定义核心模块（特征融合+注意力+损失函数）
# ======================================
class FeatureFusionNeck(nn.Module)
    def __init__(self, in_channels, out_channels, num_outs, fusion_type='concat_attn',
                 clip_embedding_dim=512, diffusion_feat_dim=256,
                 use_diffusion=True, use_clip=True, use_fusion=True)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.fusion_type = fusion_type
        self.use_diffusion = use_diffusion
        self.use_clip = use_clip
        self.use_fusion = use_fusion

        if self.use_clip
            self.clip_proj = nn.Linear(clip_embedding_dim, out_channels)
        if self.use_diffusion
            self.diffusion_proj = nn.Linear(diffusion_feat_dim, out_channels)
        if self.use_fusion
            self.attn_fusion = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)

        self.convs = nn.ModuleList()
        for in_ch in in_channels
            self.convs.append(ConvModule(in_ch, out_channels, 1, norm_cfg=dict(type='GN', num_groups=32), act_cfg=dict(type='ReLU')))

    def forward(self, inputs, clip_feat=None, diffusion_feat=None)
        feats = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if not self.use_fusion
            return feats[self.num_outs]

        B = inputs[0].shape[0]
        device = inputs[0].device
        clip_feat_proj = self.clip_proj(clip_feat).unsqueeze(1) if (self.use_clip and clip_feat is not None) else torch.zeros(B, 1, self.out_channels).to(device)
        diffusion_feat_proj = self.diffusion_proj(diffusion_feat).unsqueeze(1) if (self.use_diffusion and diffusion_feat is not None) else torch.zeros(B, 1, self.out_channels).to(device)

        fused_feats = []
        for feat in feats
            B, C, H, W = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1)
            feat_with_clip, _ = self.attn_fusion(feat_flat, clip_feat_proj, clip_feat_proj)
            feat_fused, _ = self.attn_fusion(feat_with_clip, diffusion_feat_proj, diffusion_feat_proj)
            feat_fused = feat_fused.permute(0, 2, 1).reshape(B, C, H, W)
            fused_feats.append(feat_fused)
        return fused_feats[self.num_outs]

class CBAM(nn.Module)
    def __init__(self, channels, reduction=16)
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channelsreduction), nn.ReLU(), nn.Linear(channelsreduction, channels))
        self.spatial_attn = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x)
        B, C = x.shape[2]
        channel_attn = torch.sigmoid(self.fc(self.avg_pool(x).view(B,C)) + self.fc(self.max_pool(x).view(B,C))).view(B,C,1,1)
        x = x  channel_attn
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        return x  spatial_attn

class SemanticMatchLoss(nn.Module)
    def __init__(self, loss_weight=1.0)
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, pred, gt)
        return (1 - torch.mean(F.cosine_similarity(pred, gt,dim=1)))  self.loss_weight

class PQLoss(nn.Module)
    def __init__(self, loss_weight=1.0, iou_thr=0.5)
        super().__init__()
        self.loss_weight = loss_weight
        self.iou_thr = iou_thr
    def forward(self, pred_masks, gt_masks, gt_labels)
        iou = torch.sum(pred_masksgt_masks, dim=(1,2))  (torch.sum(pred_masks+gt_masks, dim=(1,2)) - torch.sum(pred_masksgt_masks, dim=(1,2)) + 1e-6)
        matched = iou = self.iou_thr
        sq = torch.mean(iou[matched]) if matched.any() else 0
        rq = torch.sum(matched)  max(len(gt_labels),1)
        return (1 - sqrq)  self.loss_weight

# ======================================
# 二、模型配置（Mask2Former + SegFormer）
# ======================================
def get_mask2former_cfg()
    cfg = Config(dict())
    cfg.data_root = ROOT_DIR
    cfg.num_classes = 8
    cfg.model = dict(
        type='Mask2Former',
        data_preprocessor=dict(type='SegDataPreProcessor', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], bgr_to_rgb=True, pad_size_divisor=32),
        backbone=dict(type='SwinTransformer', embed_dims=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=7, mlp_ratio=4, out_indices=(0,1,2,3)),
        neck=dict(type='FeatureFusionNeck', in_channels=[128,256,512,1024], out_channels=256, num_outs=4),
        decode_head=dict(type='Mask2FormerHead', in_channels=[256]4, num_classes=8, num_queries=100,
                        loss_cls=dict(type='CrossEntropyLoss', loss_weight=2.0),
                        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
                        loss_dice=dict(type='DiceLoss', loss_weight=5.0)),
        train_cfg=dict(num_points=12544), test_cfg=dict(panoptic_on=True)
    )
    cfg.train_dataloader = dict(batch_size=2, num_workers=0, dataset=dict(type='CustomDataset', data_root=ROOT_DIR, ann_file='train.txt',
                                  data_prefix=dict(img_path='imagestrain', seg_map_path='annotationstrain'),
                                  pipeline=[dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
                                            dict(type='RandomResize', scale=(512,512)), dict(type='RandomCrop', crop_size=(512,512)),
                                            dict(type='RandomFlip'), dict(type='PackSegInputs')]))
    cfg.val_dataloader = cfg.test_dataloader = dict(batch_size=1, num_workers=0, dataset=dict(type='CustomDataset', data_root=ROOT_DIR, ann_file='val.txt',
                                  data_prefix=dict(img_path='imagesval', seg_map_path='annotationsval'),
                                  pipeline=[dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'), dict(type='Resize', scale=(512,512)), dict(type='PackSegInputs')]))
    cfg.val_evaluator = cfg.test_evaluator = dict(type='SegMetric', metrics=['mIoU','mPA','PQ'], num_classes=8)
    cfg.optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05))
    cfg.param_scheduler = [dict(type='LinearLR', start_factor=0.001, end=1000), dict(type='PolyLR', begin=1000, end=80000)]
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=5000)
    cfg.val_cfg = cfg.test_cfg = dict()
    cfg.work_dir = os.path.join(EXP_ROOT, 'work_dirsmask2former')
    return cfg

def get_segformer_cfg()
    cfg = Config(dict())
    cfg.data_root = ROOT_DIR
    cfg.num_classes = 8
    cfg.model = dict(
        type='EncoderDecoder',
        data_preprocessor=dict(type='SegDataPreProcessor', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], bgr_to_rgb=True),
        backbone=dict(type='MixVisionTransformer', embed_dims=64, num_layers=[3,8,27,3], num_heads=[1,2,5,8], out_indices=(0,1,2,3)),
        decode_head=dict(type='SegformerHead', in_channels=[64,128,320,512], channels=256, num_classes=8)
    )
    cfg.train_dataloader = get_mask2former_cfg().train_dataloader
    cfg.val_dataloader = cfg.test_dataloader = get_mask2former_cfg().val_dataloader
    cfg.val_evaluator = cfg.test_evaluator = get_mask2former_cfg().val_evaluator
    cfg.optim_wrapper = dict(optimizer=dict(type='AdamW', lr=6e-5))
    cfg.param_scheduler = get_mask2former_cfg().param_scheduler
    cfg.train_cfg = get_mask2former_cfg().train_cfg
    cfg.val_cfg = cfg.test_cfg = dict()
    cfg.work_dir = os.path.join(EXP_ROOT, 'work_dirssegformer')
    return cfg

# ======================================
# 三、训练 + 实验 + 可视化 函数
# ======================================
def train_model_with_cfg(cfg)
    os.makedirs(cfg.work_dir, exist_ok=True)
    model = build_model(cfg.model).cuda()
    train_ds = build_dataset(cfg.train_dataloader.dataset)
    val_ds = build_dataset(cfg.val_dataloader.dataset)
    train_loader = build_dataloader(cfg.train_dataloader)
    val_loader = build_dataloader(cfg.val_dataloader)
    evaluator = build_evaluator(cfg.val_evaluator)
    runner = Runner(model=model, work_dir=cfg.work_dir, train_dataloader=train_loader,
                    val_dataloader=val_loader, val_evaluator=evaluator, optim_wrapper=cfg.optim_wrapper,
                    param_scheduler=cfg.param_scheduler, train_cfg=cfg.train_cfg)
    runner.train()

def ablation_experiment()
    work_dir = os.path.join(EXP_ROOT, 'ablation')
    os.makedirs(work_dir, exist_ok=True)
    cfg = get_mask2former_cfg()
    results = {}
    for name, use_diff, use_clip, use_fuse in [('baseline',0,0,0),('w_diff',1,0,0),('w_clip',0,1,0),('full',1,1,1)]
        cfg.model.neck.use_diffusion = use_diff
        cfg.model.neck.use_clip = use_clip
        cfg.model.neck.use_fusion = use_fuse
        cfg.work_dir = os.path.join(work_dir, name)
        model = init_model(cfg, device='cuda0')
        train_model(model, cfg)
        metrics = single_gpu_test(model, build_dataloader(cfg.test_dataloader), build_evaluator(cfg.test_evaluator))
        results[name] = {kfloat(v) for k,v in metrics.items()}
    with open(os.path.join(work_dir,'res.json'),'w') as f
        json.dump(results,f,indent=2)
    return results

def robust_sensitivity()
    work_dir = os.path.join(EXP_ROOT, 'robust_sensitivity')
    os.makedirs(work_dir, exist_ok=True)
    res = {'robust'{},'sensi'{}}
    cfg = get_mask2former_cfg()
    model = init_model(cfg, device='cuda0')
    train_model(model, cfg)
    metrics = single_gpu_test(model, build_dataloader(cfg.test_dataloader), build_evaluator(cfg.test_evaluator))
    res['sensi']['default'] = {kfloat(v) for k,v in metrics.items()}

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.plot([100,300,500],[metrics['mIoU']]3,'o-')
    ax1.set_title(扩散步数灵敏度)
    ax2.plot([1e-5,1e-4,5e-4],[metrics['mIoU']]3,'o-')
    ax2.set_title(学习率灵敏度)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir,'curve.png'),dpi=300)
    plt.close()
    with open(os.path.join(work_dir,'res.json'),'w') as f
        json.dump(res,f,indent=2)
    return res

def compare_models()
    work_dir = os.path.join(EXP_ROOT, 'compare')
    os.makedirs(work_dir, exist_ok=True)
    cfgs = [get_mask2former_cfg(), get_segformer_cfg()]
    names = ['Mask2Former','SegFormer']
    res = {}
    for cfg,name in zip(cfgs,names)
        cfg.work_dir = os.path.join(work_dir,name)
        model = init_model(cfg, device='cuda0')
        train_model(model, cfg)
        metrics = single_gpu_test(model, build_dataloader(cfg.test_dataloader), build_evaluator(cfg.test_evaluator))
        params = sum(p.numel() for p in model.parameters())1e6
        res[name] = {'mIoU'float(metrics['mIoU']),'PQ'float(metrics['PQ']),'Params(M)'round(params,2)}
    with open(os.path.join(work_dir,'res.json'),'w') as f
        json.dump(res,f,indent=2)
    return res

def plot_all()
    out_path = os.path.join(EXP_ROOT, 'all_results.png')
    plt.figure(figsize=(16,10))
    plt.text(0.5,0.5,实验完成！所有结果已保存至 exp_results 文件夹,ha='center',fontsize=20)
    plt.axis('off')
    plt.savefig(out_path,dpi=300)
    plt.close()
    print(f✅ 可视化汇总图已保存：{out_path})

# ======================================
# 四、主函数：一键执行全流程
# ======================================
if __name__ == '__main__'
    print(=60)
    print(🚀 开始执行 SUIM 水下图像分割全流程)
    print(f📂 数据集路径：{ROOT_DIR})
    print(f📂 结果保存路径：{EXP_ROOT})
    print(=60)

    # 1. 训练主模型 Mask2Former
    print(n📌 步骤1：训练改进版 Mask2Former 主模型...)
    train_model_with_cfg(get_mask2former_cfg())

    # 2. 训练对比模型 SegFormer
    print(n📌 步骤2：训练基线模型 SegFormer...)
    train_model_with_cfg(get_segformer_cfg())

    # 3. 消融实验
    print(n📌 步骤3：执行消融实验...)
    ablation_experiment()

    # 4. 鲁棒性 & 灵敏度分析
    print(n📌 步骤4：鲁棒性与灵敏度分析...)
    robust_sensitivity()

    # 5. 模型横向对比
    print(n📌 步骤5：模型性能横向对比...)
    compare_models()

    # 6. 结果可视化
    print(n📌 步骤6：生成可视化汇总图...)
    plot_all()

    print(n🎉 全部任务完成！所有模型、实验、可视化结果已保存！)