# -*- coding: utf-8 -*-
"""
轻量级SNN-YOLOv8s网络定义
使用深度可分离卷积和简化的特征校正模块
"""

import torch
import torch.nn as nn
from .model import BaseModel
from ..nn.modules.yolo_spikformer import (
    MS_GetT, MS_DownSampling, MS_ConvBlock_DW, MS_AllConvBlock,
    SpikeSPPF, MS_CorrectionConv_Lightweight, MS_FusionAllConvBlock,
    MS_FusionAllConvBlock_DW, MS_Upsample, MS_DualConcat, SpikeDetect
)

# 简化的日志函数
def get_rank():
    """获取当前进程rank，简化版本返回-1"""
    return -1

class LOGGER:
    """简化的日志类"""
    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}")
    
    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}")


class SNNBackbone_Light(nn.Module):
    """
    轻量级SNN Backbone
    使用深度可分离卷积减少参数量
    - 使用MS_ConvBlock_DW替代MS_ConvBlock（参数量减少约88%）
    - mlp_ratio统一为2（进一步减少参数量）
    - width从0.5降到0.33（参数量减少约30-45%）
    """
    
    def __init__(self, img_ch=3, evs_ch=2, width=0.33, depth=0.33, max_channels=1024, verbose=False, snn_T=2, snn_img_T=None):
        super().__init__()
        
        self.verbose = verbose
        self.snn_T = snn_T  # SNN时间步数（向后兼容参数）
        # 如果未指定snn_img_T，使用snn_T（向后兼容）
        # 建议：设置snn_img_T与Events的时间步数一致（例如5）
        self.snn_img_T = snn_img_T if snn_img_T is not None else snn_T
        
        # 通道数计算函数
        def calc_channels(c, w=width, mc=max_channels):
            return min(int(c * w), mc)
        
        self.calc_channels = calc_channels
        
        # ============ Stem (输入处理) ============
        # MS_GetT的参数用于兼容性，实际不使用in_channels和out_channels
        # 使用snn_img_T来控制RGB的时间步重复次数
        self.stem = nn.ModuleDict({
            'time_convert': MS_GetT(evs_ch, evs_ch, self.snn_img_T),  # T=snn_img_T (RGB时间步数)
            'downsample': MS_DownSampling(img_in_channels=img_ch, evs_in_channels=evs_ch, embed_dims=calc_channels(128), kernel_size=7, stride=4, padding=2, first_layer=True),
            'conv_block': nn.Sequential(*[MS_ConvBlock_DW(calc_channels(128), 2, 7) for _ in range(1)])  # 使用MS_ConvBlock_DW，mlp_ratio=2
        })
        
        # ============ Stage 1 (P2) ============
        # 经过stem后，img和evs的通道数已经统一为embed_dims
        stage1_in_channels = calc_channels(128)
        stage1_out_channels = calc_channels(256)
        self.stage1 = nn.ModuleDict({
            'downsample': MS_DownSampling(img_in_channels=stage1_in_channels, evs_in_channels=stage1_in_channels, embed_dims=stage1_out_channels, kernel_size=3, stride=2, padding=1, first_layer=False),
            'conv_blocks': nn.Sequential(*[MS_ConvBlock_DW(stage1_out_channels, 2, 7) for _ in range(3)])  # 使用MS_ConvBlock_DW，mlp_ratio=2
        })
        
        # ============ Stage 2 (P3) ============
        stage2_in_channels = calc_channels(256)
        stage2_out_channels = calc_channels(512)
        self.stage2 = nn.ModuleDict({
            'downsample': MS_DownSampling(img_in_channels=stage2_in_channels, evs_in_channels=stage2_in_channels, embed_dims=stage2_out_channels, kernel_size=3, stride=2, padding=1, first_layer=False),
            'conv_blocks': nn.Sequential(*[MS_ConvBlock_DW(stage2_out_channels, 2, 7) for _ in range(4)])  # 使用MS_ConvBlock_DW，mlp_ratio=2
        })
        
        # ============ Stage 3 (P4) ============
        stage3_in_channels = calc_channels(512)
        stage3_out_channels = calc_channels(1024)
        self.stage3 = nn.ModuleDict({
            'downsample': MS_DownSampling(img_in_channels=stage3_in_channels, evs_in_channels=stage3_in_channels, embed_dims=stage3_out_channels, kernel_size=3, stride=2, padding=1, first_layer=False),
            'conv_blocks': nn.Sequential(*[MS_ConvBlock_DW(stage3_out_channels, 2, 7) for _ in range(1)])  # 使用MS_ConvBlock_DW，mlp_ratio=2（保持较小值）
        })
        
        # ============ Stage 4 (P5) ============
        self.stage4 = nn.ModuleDict({
            'sppf': SpikeSPPF(calc_channels(1024), calc_channels(1024), 5)
        })
        
        if verbose and get_rank() in (-1, 0):
            self._print_structure()
        
    def forward(self, x):
        if not isinstance(x, (tuple, list)):
            img = x
            # 如果只有img输入，创建默认的evs（2通道，snn_T个时间步）
            evs = torch.zeros(img.shape[0], self.snn_T, 2, *img.shape[2:], device=img.device)  # T=snn_T, ch=2
            x = (img, evs)
        
        # Stem: 3->64, 640->160
        x = self.stem['time_convert'](x)
        x = self.stem['downsample'](x)
        x = self.stem['conv_block'](x)
        
        # Stage 1: 64->128, 160->80 (P3)
        x = self.stage1['downsample'](x)
        p3 = self.stage1['conv_blocks'](x)
        
        # Stage 2: 128->256, 80->40 (P4)
        x = self.stage2['downsample'](p3)
        p4 = self.stage2['conv_blocks'](x)
        
        # Stage 3: 256->512, 40->20
        x = self.stage3['downsample'](p4)
        x = self.stage3['conv_blocks'](x)
        
        # Stage 4: SPPF (P5)
        p5 = self.stage4['sppf'](x)
        
        return p3, p4, p5
    
    def _print_structure(self):
        LOGGER.info("=" * 60)
        LOGGER.info("SNNBackbone_Light Structure:")
        LOGGER.info("=" * 60)
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            LOGGER.info(f"  {name:15s}: {n_params:>12,} params")
        LOGGER.info("=" * 60)


class SNNNeck_Light(nn.Module):
    """
    轻量级SNN Neck (FPN)
    使用轻量级的MS_CorrectionConv
    """
    
    def __init__(self, width=0.33, max_channels=1024, verbose=False, time_steps=2):
        super().__init__()
        
        self.verbose = verbose
        self.time_steps = time_steps
        
        # 通道数计算函数
        def calc_channels(c, w=width, mc=max_channels):
            return min(int(c * w), mc)
        
        # ============ P5分支 ============
        # 输入: calc_channels(1024) (来自backbone, width=0.5时为512)
        # 输出: calc_channels(1024) (保持通道数)
        self.p5_branch = nn.ModuleDict({
            'correction': MS_CorrectionConv_Lightweight(calc_channels(1024), calc_channels(1024), 1, 1),
            'fusion': MS_FusionAllConvBlock_DW(calc_channels(1024), 2, 7, T=self.time_steps)  # mlp_ratio=2, 深度可分离卷积版本
        })
        
        # ============ P4分支 ============
        # 输入: calc_channels(512) (来自backbone) + 上采样的calc_channels(1024) (来自P5)
        # 输出: calc_channels(512)
        self.p4_branch = nn.ModuleDict({
            'upsample': MS_Upsample(None, (1, 2, 2), 'nearest'),
            'conv': MS_ConvBlock_DW(calc_channels(1024), 2, 7),  # 使用MS_ConvBlock_DW，mlp_ratio=2
            'concat': MS_DualConcat(2),
            'correction': MS_CorrectionConv_Lightweight(calc_channels(1024) + calc_channels(512), calc_channels(512), 1, 1),
            'fusion': MS_FusionAllConvBlock_DW(calc_channels(512), 2, 7, T=self.time_steps)  # mlp_ratio=2, 深度可分离卷积版本
        })
        
        # ============ P3分支 ============
        # 输入: calc_channels(256) (来自backbone) + 上采样的calc_channels(512) (来自P4)
        # 输出: calc_channels(256)
        self.p3_branch = nn.ModuleDict({
            'upsample': MS_Upsample(None, (1, 2, 2), 'nearest'),
            'conv': MS_ConvBlock_DW(calc_channels(512), 2, 7),  # 使用MS_ConvBlock_DW，mlp_ratio=2
            'concat': MS_DualConcat(2),
            'correction': MS_CorrectionConv_Lightweight(calc_channels(512) + calc_channels(256), calc_channels(256), 1, 1),
            'fusion': MS_FusionAllConvBlock_DW(calc_channels(256), 2, 7, T=self.time_steps)  # mlp_ratio=2, 深度可分离卷积版本
        })
        
        if verbose and get_rank() in (-1, 0):
            self._print_structure()
    
    def forward(self, p3_in, p4_in, p5_in):
        # ========== P5分支 ==========
        x_p5 = self.p5_branch['correction'](p5_in)
        p5_out = self.p5_branch['fusion'](x_p5)
        
        # ========== P4分支 ==========
        x = self.p4_branch['upsample'](x_p5)
        x = self.p4_branch['conv'](x)
        x = self.p4_branch['concat']([x, p4_in])
        x_p4 = self.p4_branch['correction'](x)
        p4_out = self.p4_branch['fusion'](x_p4)
        
        # ========== P3分支 ==========
        x = self.p3_branch['upsample'](x_p4)
        x = self.p3_branch['conv'](x)
        x = self.p3_branch['concat']([x, p3_in])
        x_p3 = self.p3_branch['correction'](x)
        p3_out = self.p3_branch['fusion'](x_p3)
        
        return p3_out, p4_out, p5_out
    
    def _print_structure(self):
        LOGGER.info("=" * 60)
        LOGGER.info("SNNNeck_Light Structure:")
        LOGGER.info("=" * 60)
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            LOGGER.info(f"  {name:15s}: {n_params:>12,} params")
        LOGGER.info("=" * 60)


class SNNDetectionHead_Light(nn.Module):
    """SNN Detection Head"""
    
    def __init__(self, nc=8, channels=[256, 128, 128], verbose=False):
        super().__init__()
        
        self.nc = nc
        self.detect = SpikeDetect(nc, channels)
        
        if verbose and get_rank() in (-1, 0):
            n_params = sum(p.numel() for p in self.detect.parameters())
            LOGGER.info("=" * 60)
            LOGGER.info(f"SNNDetectionHead_Light: {n_params:,} params")
            LOGGER.info(f"  Classes: {nc}")
            LOGGER.info(f"  Input channels: {channels}")
            LOGGER.info("=" * 60)
    
    def forward(self, p3, p4, p5):
        return self.detect([p5, p4, p3])


class SNNYOLOv8s_Light(BaseModel):
    """
    轻量级SNN-YOLOv8s网络
    
    优化:
    - 使用MS_ConvBlock_DW（深度可分离卷积，参数量减少约88%）
    - mlp_ratio统一为2（进一步减少参数量）
    - width从0.5降到0.33（参数量减少约30-45%）
    - 简化FeatureCorrection模块（reduction=4）
    
    预期效果:
    - 参数量减少 50-70%
    - 推理速度提升 20-30%
    - 精度损失 3-8%
    """
    
    def __init__(self, nc=8, img_ch=3, evs_ch=2, verbose=True):
        super().__init__()
        
        # Scale parameters for 's' version
        self.depth = 0.33
        self.width = 0.33
        self.max_channels = 1024
        
        # 通道数计算函数
        def calc_channels(c):
            return min(int(c * self.width), self.max_channels)
        
        if verbose and get_rank() in (-1, 0):
            LOGGER.info("\n" + "=" * 70)
            LOGGER.info("Building Lightweight SNN-YOLOv8s Network")
            LOGGER.info("=" * 70)
            LOGGER.info(f"Scale: 's' (depth={self.depth}, width={self.width})")
            LOGGER.info(f"Optimizations: Depthwise Separable Conv + Lightweight Correction")
            LOGGER.info(f"Classes: {nc}, Image channels: {img_ch}, Event channels: {evs_ch}")
            LOGGER.info("=" * 70)
        
        # ============ 1. Backbone ============
        if verbose and get_rank() in (-1, 0):
            LOGGER.info("\n[1/3] Building Lightweight Backbone...")
        self.backbone = SNNBackbone_Light(
            img_ch=img_ch,
            evs_ch=evs_ch,
            width=self.width,
            depth=self.depth,
            max_channels=self.max_channels,
            verbose=verbose
        )
        
        # ============ 2. Neck ============
        if verbose and get_rank() in (-1, 0):
            LOGGER.info("\n[2/3] Building Lightweight Neck...")
        self.neck = SNNNeck_Light(
            width=self.width,
            max_channels=self.max_channels,
            verbose=verbose
        )
        
        # ============ 3. Head ============
        if verbose and get_rank() in (-1, 0):
            LOGGER.info("\n[3/3] Building Detection Head...")
        
        detect_channels = [
            calc_channels(512),  # P5: 256
            calc_channels(256),  # P4: 128
            calc_channels(256)   # P3: 128
        ]
        
        self.head = SNNDetectionHead_Light(
            nc=nc,
            channels=detect_channels,
            verbose=verbose
        )
        
        # 存储模型信息
        self.nc = nc
        self.names = {i: f'{i}' for i in range(nc)}
        self.inplace = True
        
        # 【修复】设置stride顺序与feats顺序匹配
        # Head传递顺序是[p5(20x20), p4(40x40), p3(80x80)]
        # 所以stride应该是[32, 16, 8]而不是[8, 16, 32]
        self.head.detect.stride = torch.tensor([32.0, 16.0, 8.0])
        self.stride = self.head.detect.stride
        self.head.detect.inplace = self.inplace  # type: ignore
        self.head.detect.bias_init()
        
        # 初始化权重
        self.initialize_weights()
        
        if verbose and get_rank() in (-1, 0):
            self._print_summary()
    
    def forward(self, x):
        # Backbone
        p3, p4, p5 = self.backbone(x)
        
        # Neck
        p3_fused, p4_fused, p5_fused = self.neck(p3, p4, p5)
        
        # Head
        return self.head(p3_fused, p4_fused, p5_fused)
    
    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3  # type: ignore
                m.momentum = 0.03  # type: ignore
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # type: ignore
    
    def init_criterion(self):
        """初始化损失函数 - 网络结构文件中不包含损失函数实现"""
        raise NotImplementedError("Loss function should be implemented in training code, not in network architecture")
    
    def _print_summary(self):
        """打印模型总结"""
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("Lightweight SNN-YOLOv8s Summary")
        LOGGER.info("=" * 70)
        LOGGER.info(f"{'Module':<20s} {'Parameters':>15s} {'Percentage':>12s}")
        LOGGER.info("-" * 70)
        LOGGER.info(f"{'Backbone':<20s} {backbone_params:>15,} {backbone_params/total_params*100:>11.1f}%")
        LOGGER.info(f"{'Neck (FPN)':<20s} {neck_params:>15,} {neck_params/total_params*100:>11.1f}%")
        LOGGER.info(f"{'Head (Detect)':<20s} {head_params:>15,} {head_params/total_params*100:>11.1f}%")
        LOGGER.info("-" * 70)
        LOGGER.info(f"{'Total':<20s} {total_params:>15,} {'100.0%':>12s}")
        LOGGER.info("=" * 70)


if __name__ == '__main__':
    print("=" * 70)
    print("  测试轻量级SNN-YOLOv8s")
    print("=" * 70)
    
    # 创建模型
    print("\n创建轻量级模型...")
    model = SNNYOLOv8s_Light(nc=8, img_ch=3, evs_ch=2, verbose=True)
    
    # 打印参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\n轻量级模型参数量: {params:,}")
    
    # 注意：网络结构文件中不包含原版模型的对比代码
    
    # 测试前向传播
    print("\n测试前向传播...")
    model.eval()
    x_img = torch.randn(1, 3, 640, 640)  # RGB图像，3通道
    x_evs = torch.randn(1, 5, 2, 640, 640)  # 事件数据，T=5, ch=2
    
    with torch.no_grad():
        import time
        start = time.time()
        output = model((x_img, x_evs))
        forward_time = (time.time() - start) * 1000
    
    print(f"✓ 前向传播成功！耗时: {forward_time:.2f}ms")
    print(f"✓ 输出形状: {output[0].shape}")
    
    print("\n" + "=" * 70)
    print("✅ 轻量级模型测试完成！")
    print("=" * 70)
