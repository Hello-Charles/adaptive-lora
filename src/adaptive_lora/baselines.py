"""M6 三种标准 LoRA 的层集合规划；不重算重要性、不训练，也不把层数比例当实测参数量。

Full 与 K-layer 使用同一 rank/投影规格，但参数预算不同。Random-K 的局部 RNG 只负责
无放回均匀选层，不改变调用者的训练随机状态；正式比较必须另存训练 seed 和选层 seed。
"""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BaselinePlan:
    """零基层号不可变快照；indices 保留 Full 顺序、Random 抽取顺序或 Adaptive 重要性顺序。

    定位/注入时可以升序，但不能用排序后的编号冒充原始随机抽取或重要性排名。
    selection_seed 仅对 Random-K 有意义，其他方法为 None；不由该字段控制训练 RNG。
    """

    method: str
    layer_count: int
    indices: tuple[int, ...]
    selection_seed: int | None = None

    def to_dict(self):
        """返回独立可序列化记录；列表修改不影响计划本身。"""
        return {"method": self.method, "index_base": 0, "layer_count": self.layer_count,
                "selected_layer_count": len(self.indices), "selected_layer_indices": list(self.indices),
                "selected_layer_indices_ascending": sorted(self.indices), "selection_seed": self.selection_seed}


def plan_baselines(layer_count, k, adaptive_indices, *, layer_seed):
    """返回固定顺序的 Full / Random-K / Adaptive-K 计划，不从评估结果改变层集合。

    所有数量/索引/seed 要求普通整数，拒绝 bool、重复/缺失/越界 Adaptive 层。调用者须
    先核验 Adaptive 输入来自冻结基础模型的训练侧校准；本函数不能证明其数据来源。
    均匀抽样由独立 random.Random(seed).sample 实现，重复性还依赖 Python 版本。
    """
    if (type(layer_count) is not int or layer_count < 1 or type(k) is not int or not 1 <= k <= layer_count
            or type(layer_seed) is not int or not 0 <= layer_seed < 2 ** 32):
        raise ValueError("Invalid layer_count, k or layer selection seed")
    adaptive = tuple(adaptive_indices)
    if (len(adaptive) != k or any(type(index) is not int or not 0 <= index < layer_count for index in adaptive)
            or len(set(adaptive)) != k):
        raise ValueError("Adaptive indices must be exactly K unique, legal zero-based integers")
    sampled = tuple(random.Random(layer_seed).sample(range(layer_count), k))
    return (BaselinePlan("full", layer_count, tuple(range(layer_count))),
            BaselinePlan("random_k", layer_count, sampled, layer_seed),
            BaselinePlan("adaptive_k", layer_count, adaptive))
