"""基于校准重要性的全层排名与 Top-K；不计算分数、不创建 LoRA 或启动训练。"""

import math
from numbers import Real


def rank_layers(scores):
    """按分数降序返回零基层索引；相等分数按索引升序，输入数组不被修改。

    scores 必须非空、有限且非负，下标就是层号，不接受字符串/布尔值等隐式数值转换。
    并列规则属于 [复现决策]，与论文 Top-K 不冲突，但不能声称来自作者代码。
    """
    scores = list(scores)
    if not scores or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0
        for value in scores
    ):
        raise ValueError("scores must be non-empty finite non-negative real numbers")
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def select_top_k(scores, k):
    """返回恰好 k 个合法唯一层索引，保留重要性排名顺序而非按层号重新排序。

    不会静默裁剪超范围 k；调用者要按层号显示时应另行排序，不能丢失原始重要性名次。
    """
    order = rank_layers(scores)
    if type(k) is not int or not 1 <= k <= len(order):
        raise ValueError("k must be an integer between 1 and the number of layers")
    return order[:k]
