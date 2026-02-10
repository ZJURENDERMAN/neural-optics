"""
损失函数模块 - 简化版

提供光学优化常用的损失函数
"""

import numpy as np
from typing import Dict, Any, List


class LossModule:
    """损失函数基类"""
    
    def __init__(self, weight: float = 1.0, name: str = "Loss"):
        self.weight = weight
        self.name = name
        self._dr = None
        self._Float = None
    
    def set_backend(self, dr, Float):
        self._dr = dr
        self._Float = Float
    
    def compute(self, pred, target):
        raise NotImplementedError
    
    def __call__(self, pred, target):
        return self.weight * self.compute(pred, target)


class L1Loss(LossModule):
    """L1 损失（平均绝对误差）"""
    
    def __init__(self, weight: float = 1.0, normalize: bool = True):
        super().__init__(weight, "L1")
        self.normalize = normalize
    
    def compute(self, pred, target):
        dr = self._dr
        if self.normalize:
            pred_sum = dr.sum(dr.detach(pred)) + 1e-10
            target_sum = dr.sum(dr.detach(target)) + 1e-10
            diff = dr.abs(pred / pred_sum - target / target_sum)
        else:
            diff = dr.abs(pred - target)
        return dr.mean(diff)


class L2Loss(LossModule):
    """L2 损失（均方误差）"""
    
    def __init__(self, weight: float = 1.0, normalize: bool = True):
        super().__init__(weight, "L2")
        self.normalize = normalize
    
    def compute(self, pred, target):
        dr = self._dr
        if self.normalize:
            pred_sum = dr.sum(dr.detach(pred)) + 1e-10
            target_sum = dr.sum(dr.detach(target)) + 1e-10
            diff = pred / pred_sum - target / target_sum
        else:
            diff = pred - target
        return dr.mean(diff * diff)


class SmoothL1Loss(LossModule):
    """Smooth L1 损失（Huber损失）"""
    
    def __init__(self, weight: float = 1.0, beta: float = 0.1, normalize: bool = True):
        super().__init__(weight, "SmoothL1")
        self.beta = beta
        self.normalize = normalize
    
    def compute(self, pred, target):
        dr = self._dr
        Float = self._Float
        
        if self.normalize:
            pred_sum = dr.sum(dr.detach(pred)) + 1e-10
            target_sum = dr.sum(dr.detach(target)) + 1e-10
            diff = dr.abs(pred / pred_sum - target / target_sum)
        else:
            diff = dr.abs(pred - target)
        
        beta = Float(self.beta)
        loss = dr.select(diff < beta, 
                        0.5 * diff * diff / beta, 
                        diff - 0.5 * beta)
        return dr.mean(loss)


class BoundaryPenaltyLoss(LossModule):
    """
    边界惩罚损失
    
    惩罚落在目标为0区域的光线，帮助收敛边缘散逸的光线
    """
    
    def __init__(self, weight: float = 1.0, threshold: float = 0.05, 
                 width: int = 256, height: int = 256):
        super().__init__(weight, "BoundaryPenalty")
        self.threshold = threshold
        self.width = width
        self.height = height
    
    def compute(self, pred, target):
        dr = self._dr
        Float = self._Float
        
        # 创建边界掩码：目标值低于阈值的区域
        target_np = np.array(target).reshape(self.height, self.width)
        target_max = target_np.max()
        if target_max > 0:
            target_normalized = target_np / target_max
        else:
            target_normalized = target_np
        
        # 低于阈值 = 应该为0的区域
        boundary_mask_np = (target_normalized < self.threshold).astype(np.float32)
        boundary_mask = Float(boundary_mask_np.flatten())
        
        # 归一化预测
        pred_sum = dr.sum(dr.detach(pred)) + 1e-10
        pred_norm = pred / pred_sum
        
        # 惩罚在边界区域的光线
        penalty = boundary_mask * pred_norm
        return dr.sum(penalty)


class ConcentrationLoss(LossModule):
    """
    集中度损失
    
    鼓励光线集中在目标非零区域
    """
    
    def __init__(self, weight: float = 1.0, width: int = 256, height: int = 256):
        super().__init__(weight, "Concentration")
        self.width = width
        self.height = height
    
    def compute(self, pred, target):
        dr = self._dr
        Float = self._Float
        
        # 归一化
        pred_sum = dr.sum(dr.detach(pred)) + 1e-10
        pred_norm = pred / pred_sum
        
        # 创建反向权重：目标为0的地方权重高
        target_np = np.array(target).reshape(self.height, self.width)
        target_max = target_np.max() + 1e-10
        inverse_weight = 1.0 - (target_np / target_max)
        inverse_weight = Float(inverse_weight.flatten().astype(np.float32))
        
        # 加权惩罚
        weighted_penalty = inverse_weight * pred_norm
        return dr.sum(weighted_penalty)


class GradientLoss(LossModule):
    """梯度损失 - 保持边缘锐度"""
    
    def __init__(self, weight: float = 1.0, width: int = 256, height: int = 256):
        super().__init__(weight, "Gradient")
        self.width = width
        self.height = height
    
    def compute(self, pred, target):
        dr = self._dr
        Float = self._Float
        
        pred_np = np.array(pred).reshape(self.height, self.width)
        target_np = np.array(target).reshape(self.height, self.width)
        
        # 归一化
        pred_np = pred_np / (pred_np.sum() + 1e-10)
        target_np = target_np / (target_np.sum() + 1e-10)
        
        # 计算梯度差异
        pred_gx = np.diff(pred_np, axis=1)
        pred_gy = np.diff(pred_np, axis=0)
        target_gx = np.diff(target_np, axis=1)
        target_gy = np.diff(target_np, axis=0)
        
        loss = np.mean(np.abs(pred_gx - target_gx)) + np.mean(np.abs(pred_gy - target_gy))
        return Float([loss])[0]


class CombinedLoss(LossModule):
    """组合多个损失函数"""
    
    def __init__(self, losses: List[LossModule] = None):
        super().__init__(1.0, "Combined")
        self.losses = losses or []
    
    def add_loss(self, loss: LossModule):
        self.losses.append(loss)
        return self
    
    def set_backend(self, dr, Float):
        super().set_backend(dr, Float)
        for loss in self.losses:
            loss.set_backend(dr, Float)
    
    def compute(self, pred, target):
        total_loss = self._Float([0.0])[0]
        for loss in self.losses:
            total_loss = total_loss + loss(pred, target)
        return total_loss


class LossFactory:
    """损失函数工厂"""
    
    @staticmethod
    def create_loss(config: Dict[str, Any], dr, Float, 
                    width: int = 256, height: int = 256) -> LossModule:
        loss_type = config.get("type", "l1").lower()
        weight = config.get("weight", 1.0)
        
        if loss_type == "l1":
            loss = L1Loss(weight=weight, normalize=config.get("normalize", True))
        
        elif loss_type == "l2":
            loss = L2Loss(weight=weight, normalize=config.get("normalize", True))
        
        elif loss_type == "smooth_l1" or loss_type == "huber":
            loss = SmoothL1Loss(
                weight=weight,
                beta=config.get("beta", 0.1),
                normalize=config.get("normalize", True)
            )
        
        elif loss_type == "boundary_penalty" or loss_type == "boundary":
            loss = BoundaryPenaltyLoss(
                weight=weight,
                threshold=config.get("threshold", 0.05),
                width=width,
                height=height
            )
        
        elif loss_type == "concentration":
            loss = ConcentrationLoss(
                weight=weight,
                width=width,
                height=height
            )
        
        elif loss_type == "gradient":
            loss = GradientLoss(
                weight=weight,
                width=width,
                height=height
            )
        
        else:
            raise ValueError(f"未知的损失类型: {loss_type}")
        
        loss.set_backend(dr, Float)
        return loss
    
    @staticmethod
    def create_combined_loss(configs: List[Dict[str, Any]], dr, Float,
                             width: int = 256, height: int = 256) -> CombinedLoss:
        combined = CombinedLoss()
        for cfg in configs:
            loss = LossFactory.create_loss(cfg, dr, Float, width, height)
            combined.add_loss(loss)
        combined.set_backend(dr, Float)
        return combined
    
    @staticmethod
    def from_json_config(loss_config: Dict[str, Any], dr, Float,
                         width: int = 256, height: int = 256) -> LossModule:
        if "losses" in loss_config:
            return LossFactory.create_combined_loss(
                loss_config["losses"], dr, Float, width, height
            )
        else:
            return LossFactory.create_loss(loss_config, dr, Float, width, height)