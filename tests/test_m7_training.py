"""M7 训练执行层的离线测试：设备/精度解析、累积组切分与真实梯度数学；不下载、不用 GPU。

测试能力边界：

- `run_accumulation` 的数学断言使用极小 CPU 线性模型，验证的是**累积语义**（组内均值、
  只在组末步进、unscale 后检查），不是 Qwen 前向或论文训练。
- CUDA 分支只在 `torch.cuda.is_available()` 为假的环境中断言"报错而不回退"，无法在此
  证明真实 CUDA/AMP 行为；GPU 上的数值与显存仍需独立验收。
- 本文件不加载 tokenizer、不读数据、不写 checkpoint。
"""

import math
import unittest

import torch

from adaptive_lora.m7_training import (
    CONFIRMED_FINAL_PARTIAL_GROUP,
    FINAL_GROUP_POLICIES,
    describe_batch_plan,
    plan_groups,
    resolve_amp,
    resolve_device,
    run_accumulation,
)


def tiny_model():
    """返回单层线性模型与它的两个可训练参数；固定初值使断言可重复。"""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2, bias=True)
    # 显式设为已知值，避免依赖 torch 初始化实现的稳定性。
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.1]]))
        model.bias.copy_(torch.tensor([0.05, -0.05]))
    return model


def make_forward_loss(model, micro_batches):
    """返回**调度器**：接收 micro-batch 序号并直接算出该样本的标量 loss。

    注意区分两个层次：本函数返回的函数才是 `run_accumulation` 所需的 `forward_loss`
    签名（micro -> 标量张量）。内部 `sample_loss(index)` 用于构造参考梯度时单独求值。
    """
    def sample_loss(index):
        inputs, target = micro_batches[index]
        return torch.nn.functional.mse_loss(model(inputs), target)

    def forward_loss(micro):
        # micro 是调用方传入的 micro-batch 标识；此处即为其在列表中的序号。
        return sample_loss(int(micro))

    forward_loss.sample_loss = sample_loss
    return forward_loss


class DeviceTests(unittest.TestCase):
    """设备解析必须显式失败，不能把 GPU 请求静默降级为 CPU。"""

    def test_cpu_resolves(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_cuda_unavailable_raises_instead_of_falling_back(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available; the no-fallback path cannot be exercised here")
        with self.assertRaisesRegex(ValueError, "refusing to silently fall back"):
            resolve_device("cuda")

    def test_invalid_device_rejected(self):
        for value in ("gpu", "cuda:0", "", None, 0, "CPU"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_device(value)


class AmpTests(unittest.TestCase):
    """CPU 上必须显式禁用 AMP；不把论文的 CUDA FP16 AMP 说成已生效。"""

    def test_cpu_amp_is_disabled_and_reported(self):
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        self.assertFalse(autocast["enabled"])
        self.assertEqual(autocast["device_type"], "cpu")
        self.assertEqual(autocast["dtype"], torch.float16)
        self.assertFalse(scaler.is_enabled())

    def test_cpu_bfloat16_also_disabled(self):
        autocast, _ = resolve_amp(torch.device("cpu"), "bfloat16")
        self.assertFalse(autocast["enabled"])
        self.assertEqual(autocast["dtype"], torch.bfloat16)

    def test_disabled_by_caller_flag(self):
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16", enabled=False)
        self.assertFalse(autocast["enabled"])
        self.assertFalse(scaler.is_enabled())

    def test_unsupported_dtype_and_non_bool_flag_rejected(self):
        for dtype in ("float32", "fp16", "", None, "bfloat16 "):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                resolve_amp(torch.device("cpu"), dtype)
        with self.assertRaises(ValueError):
            resolve_amp(torch.device("cpu"), "float16", enabled="yes")


class GroupPlanTests(unittest.TestCase):
    """累积组切分必须在任何前向之前确定，且不静默改变有效 batch。"""

    def test_even_split(self):
        self.assertEqual(plan_groups(16, 8), [8, 8])
        self.assertEqual(plan_groups(0, 8), [])

    def test_error_policy_rejects_remainder(self):
        with self.assertRaisesRegex(ValueError, "do not divide into groups"):
            plan_groups(1250, 8, "error")

    def test_drop_policy_confirmed_discards_leftover(self):
        """已确认决策：drop 丢弃末尾不足一组的 micro-batch，而不是报错。

        `plan_groups` 的返回值只覆盖完整组，因此调用方必须用 `count - sum(groups)`
        自行记账；本测试把这个数字钉死在论文形状上（1,250 → 丢 2）。
        """
        self.assertEqual(plan_groups(10, 8, "drop"), [8])
        self.assertEqual(plan_groups(1250, 8, "drop"), [8] * 156)
        self.assertEqual(1250 - sum(plan_groups(1250, 8, "drop")), 2)
        # 已确认策略是该模块的默认值，不需要调用方额外传参。
        self.assertEqual(plan_groups(1250, 8), plan_groups(1250, 8, "drop"))
        self.assertEqual(CONFIRMED_FINAL_PARTIAL_GROUP, "drop")

    def test_step_policy_allows_smaller_final_group(self):
        self.assertEqual(plan_groups(10, 8, "step"), [8, 2])
        self.assertEqual(plan_groups(1250, 8, "step"), [8] * 156 + [2])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            plan_groups(-1, 8)
        with self.assertRaises(ValueError):
            plan_groups(8, 0)
        with self.assertRaises(ValueError):
            plan_groups(8, 8, "ignore")
        for policy in FINAL_GROUP_POLICIES:
            self.assertIn(policy, FINAL_GROUP_POLICIES)


class BatchPlanTests(unittest.TestCase):
    """论文 10,000 条训练样本与 8×8=64 不整除，这一事实必须被显式暴露。"""

    def test_paper_shape_exposes_leftover(self):
        plan = describe_batch_plan(10000, 8, 8, epochs=3)
        self.assertEqual(plan["effective_batch_size"], 64)
        self.assertEqual(plan["micro_batches_per_epoch"], 1250)
        self.assertEqual(plan["full_groups_per_epoch"], 156)
        self.assertEqual(plan["leftover_micro_batches_per_epoch"], 2)
        self.assertEqual(plan["samples_left_over_per_epoch"], 16)
        self.assertEqual(plan["samples_never_used_per_epoch"], 0)
        self.assertFalse(plan["divides_evenly"])
        # 已确认 drop：156 组完整步进，末尾 2 个 micro-batch 被丢弃。
        self.assertEqual(plan["final_partial_group_policy"], "drop")
        self.assertEqual(plan["optimizer_steps_per_epoch"], 156)
        self.assertEqual(plan["samples_used_per_epoch"], 9984)
        self.assertEqual(plan["samples_dropped_per_epoch"], 16)
        # 三个 epoch 累计：48 条样本从未参与任何权重更新，必须如实暴露。
        self.assertEqual(plan["optimizer_steps_total"], 468)
        self.assertEqual(plan["optimizer_steps_total"], 468)
        # 计数单位是"样本-次"：3 epochs × 16 条。在已确认的 training_order='none' 下，
        # 每 epoch 顺序相同，因此这 48 个"样本-次"其实是同一批 16 条样本被重复丢弃，
        # 而不是 48 条不同样本。
        self.assertEqual(plan["samples_used_total_visits"], 29952)
        self.assertEqual(plan["samples_dropped_total_visits"], 48)
        self.assertIn("same samples", plan["dropped_distinctness_note"])
        self.assertAlmostEqual(plan["samples_dropped_fraction_total"], 48 / 30000, places=12)

    def test_step_policy_uses_the_remainder_instead_of_dropping(self):
        """对照：step 策略下末尾 2 个 micro-batch 照常步进，总计 157 步、零丢弃。"""
        plan = describe_batch_plan(10000, 8, 8, epochs=1, policy="step")
        self.assertEqual(plan["optimizer_steps_per_epoch"], 157)
        self.assertEqual(plan["samples_used_per_epoch"], 10000)
        self.assertEqual(plan["samples_dropped_per_epoch"], 0)

    def test_divisible_case(self):
        plan = describe_batch_plan(9984, 8, 8, epochs=1)
        self.assertTrue(plan["divides_evenly"])
        self.assertEqual(plan["leftover_micro_batches_per_epoch"], 0)
        self.assertEqual(plan["optimizer_steps_total"], 156)
        self.assertEqual(plan["samples_dropped_total_visits"], 0)

    def test_micro_batch_remainder_is_reported_separately(self):
        # 10003 条：每 epoch 有 3 条样本连一个 micro-batch 都凑不满。
        plan = describe_batch_plan(10003, 8, 8)
        self.assertEqual(plan["samples_never_used_per_epoch"], 3)
        self.assertEqual(plan["micro_batches_per_epoch"], 1250)
        # 两层余量叠加：3 条不完整批 + 2 个被 drop 的 micro-batch（16 条）= 19 条。
        self.assertEqual(plan["samples_dropped_per_epoch"], 19)

    def test_invalid_values_rejected(self):
        for args in ((0, 8, 8), (100, 0, 8), (100, 8, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                describe_batch_plan(*args)
        # epochs 是仅限关键字参数，非法值同样必须拒绝而不是被忽略。
        for epochs in (0, -1, True, 1.0):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                describe_batch_plan(100, 8, 8, epochs=epochs)
        with self.assertRaises(ValueError):
            describe_batch_plan(100, 8, 8, policy="ignore")


class AccumulationMathTests(unittest.TestCase):
    """核心数学：累积梯度必须等于"组内平均 loss"的梯度，而不是组内求和的梯度。"""

    def _fixtures(self):
        model = tiny_model()
        batches = [
            (torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0]])),
            (torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.0, 1.0]])),
            (torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.5, 0.5]])),
            (torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([[0.25, 0.75]])),
        ]
        return model, batches

    def test_accumulated_gradient_equals_mean_over_group(self):
        """两个 micro-batch 累积一步，梯度必须等于两条 loss 平均值对参数的梯度。

        若实现误用求和（不除以组大小），梯度会恰好多一倍；这里用逐分量比较捕捉该错误。
        """
        model, batches = self._fixtures()
        factory = make_forward_loss(model, batches[:2])
        trainable = list(model.parameters())

        # 参考梯度：直接对 (loss0 + loss1) / 2 求导。
        model.zero_grad(set_to_none=True)
        reference = (factory.sample_loss(0) + factory.sample_loss(1)) / 2
        reference.backward()
        expected = [parameter.grad.detach().clone() for parameter in trainable]
        model.zero_grad(set_to_none=True)

        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")

        # 用 lr=0 的等价手段无法验证提交流程，因此改为在 step 前的回调里抓梯度。
        # run_accumulation 只在组末调用 unscale_ 后检查范数；这里通过包装 optimizer.step
        # 在其真正更新前快照梯度，update 之前梯度尚未清零。
        captured = {}

        original_step = optimizer.step

        def snapshot_step(*args, **kwargs):
            captured["grads"] = [parameter.grad.detach().clone() for parameter in trainable]
            return original_step(*args, **kwargs)

        optimizer.step = snapshot_step
        result = run_accumulation(
            optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
            forward_loss=factory, accumulation=2, device=torch.device("cpu"),
            autocast=autocast, scaler=scaler,
        )

        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["micro_batches_used"], 2)
        for index, (actual, want) in enumerate(zip(captured["grads"], expected)):
            with self.subTest(parameter=index):
                self.assertTrue(
                    torch.allclose(actual, want, rtol=1e-6, atol=1e-9),
                    f"accumulated gradient differs from the mean-loss gradient: {actual} vs {want}",
                )

    def test_loss_reported_is_unscaled_mean(self):
        """报告的 loss 必须是原始 loss 的组内均值，不受 /size 缩放影响。"""
        model, batches = self._fixtures()
        factory = make_forward_loss(model, batches[:2])
        trainable = list(model.parameters())
        expected = ((factory.sample_loss(0) + factory.sample_loss(1)) / 2).item()

        observed = []
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        run_accumulation(
            optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
            forward_loss=factory, accumulation=2, device=torch.device("cpu"),
            autocast=autocast, scaler=scaler, on_step=observed.append,
        )
        self.assertAlmostEqual(observed[0]["mean_loss"], expected, places=6)

    def test_steps_only_at_group_end(self):
        """4 个 micro-batch / 累积 2 → 只有 2 次 optimizer step，而不是 4 次。"""
        model, batches = self._fixtures()
        factory = make_forward_loss(model, batches)
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        result = run_accumulation(
            optimizer=optimizer, trainable=trainable, micro_batches=[0, 1, 2, 3],
            forward_loss=factory, accumulation=2, device=torch.device("cpu"),
            autocast=autocast, scaler=scaler,
        )
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual([step["micro_batches_in_group"] for step in result["steps"]], [2, 2])

    def test_partial_group_uses_actual_size_for_mean(self):
        """step 策略下最后一组按实际大小求均值；3 个 micro-batch / 累积 2 → [2,1]。

        用 `lr=0` 固定权重，使"参考梯度"对各组都可比：本测试验证的是累积语义，
        不是 AdamW 的更新效果。若 lr 非零，第一组已改动权重，第二组的梯度理应不同，
        那样比较参考梯度就会把优化器行为误判为累积实现错误。
        """
        model, batches = self._fixtures()
        factory = make_forward_loss(model, batches[:3])
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=0.0)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")

        captured = {}
        original_step = optimizer.step
        calls = {"n": 0}

        def snapshot_step(*args, **kwargs):
            """在 optimizer 真正更新前快照第 2 组（即最后一组）的梯度。"""
            calls["n"] += 1
            if calls["n"] == 2:
                captured["grads"] = [parameter.grad.detach().clone() for parameter in trainable]
            return original_step(*args, **kwargs)

        optimizer.step = snapshot_step
        result = run_accumulation(
            optimizer=optimizer, trainable=trainable, micro_batches=[0, 1, 2],
            forward_loss=factory, accumulation=2, device=torch.device("cpu"),
            autocast=autocast, scaler=scaler, final_partial_group="step",
        )
        self.assertEqual([step["micro_batches_in_group"] for step in result["steps"]], [2, 1])
        self.assertEqual(result["effective_batch_size_actual_last"], 1)

        # 最后一组只有一个 micro-batch，其梯度必须等于该条 loss 自身的梯度（除以 1）。
        model.zero_grad(set_to_none=True)
        factory.sample_loss(2).backward()
        expected_last = [parameter.grad.detach().clone() for parameter in trainable]
        model.zero_grad(set_to_none=True)
        for index, (actual, want) in enumerate(zip(captured["grads"], expected_last)):
            with self.subTest(parameter=index):
                self.assertTrue(
                    torch.allclose(actual, want, rtol=1e-6, atol=1e-9),
                    f"partial-group gradient differs from the single-sample gradient: {actual} vs {want}",
                )

    def test_non_divisible_with_error_policy_raises_before_any_forward(self):
        """error 策略必须在第一次前向之前失败，不能跑一半才报错。"""
        model, batches = self._fixtures()
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        calls = {"n": 0}

        def factory(_micro):
            calls["n"] += 1
            return torch.tensor(0.0, requires_grad=True)

        with self.assertRaisesRegex(ValueError, "do not divide into groups"):
            run_accumulation(
                optimizer=optimizer, trainable=trainable, micro_batches=[0, 1, 2],
                forward_loss=factory, accumulation=2, device=torch.device("cpu"),
                autocast=autocast, scaler=scaler, final_partial_group="error",
            )
        self.assertEqual(calls["n"], 0, "no forward should run when the plan is invalid")

    def test_missing_gradient_is_reported_not_crashed(self):
        """前向未连到可训练参数时必须给出明确报错，而不是在范数统计处以 TypeError 崩掉。

        这不是纯理论风险：若 loss 计算错误地漏掉 adapter（例如误用冻结的基础输出），
        梯度从未生成，`.grad is None` 一旦进入 `count_nonzero` 就会抛 TypeError，
        把"loss 没接上参数"误报成张量类型错误。
        """
        model, _ = self._fixtures()
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")

        def disconnected(_micro):
            # 常数 loss：requires_grad 为真但与被优化参数无关，因此不会产生梯度。
            return torch.tensor(1.0, requires_grad=True)

        with self.assertRaisesRegex(ValueError, "Missing gradient"):
            run_accumulation(
                optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
                forward_loss=disconnected, accumulation=2, device=torch.device("cpu"),
                autocast=autocast, scaler=scaler, final_partial_group="error",
            )
        # 失败路径也必须清空梯度，不把半成品状态留给调用方。
        self.assertTrue(all(parameter.grad is None for parameter in trainable))

    def test_drop_policy_skips_the_leftover_and_reports_it(self):
        """已确认 drop：末尾 1 个 micro-batch 既不前向也不步进，且被如实计数。

        3 个 micro-batch / 累积 2 → 只有 1 次 step（覆盖前 2 个），第 3 个被丢弃。
        关键断言是被丢弃的那个**从未进入 forward_loss**：静默多算一次前向会让"用了多少样本"
        与优化器步数对不上。
        """
        model, batches = self._fixtures()
        factory = make_forward_loss(model, batches[:3])
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")

        visited = []
        result = run_accumulation(
            optimizer=optimizer, trainable=trainable, micro_batches=[0, 1, 2],
            forward_loss=factory, accumulation=2, device=torch.device("cpu"),
            autocast=autocast, scaler=scaler, final_partial_group="drop",
            on_micro_batch=lambda record: visited.append(record["offset"]),
        )
        self.assertEqual(result["final_partial_group_policy"], "drop")
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["micro_batches_supplied"], 3)
        self.assertEqual(result["micro_batches_used"], 2)
        self.assertEqual(result["micro_batches_dropped"], 1)
        self.assertEqual(visited, [0, 1], "the dropped micro-batch must never be forwarded")
        # 默认策略即为已确认的 drop，无需显式传参。
        self.assertEqual(CONFIRMED_FINAL_PARTIAL_GROUP, "drop")


class GuardTests(unittest.TestCase):
    """形状、白名单与状态守卫；错误输入不得进入优化器。"""

    def _setup(self, accumulation=2):
        model = tiny_model()
        batches = [(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0]]))] * accumulation
        return model, batches

    def test_optimizer_whitelist_mismatch_rejected(self):
        model, batches = self._setup()
        factory = make_forward_loss(model, batches)
        # 优化器只含部分参数：必须拒绝，因为冻结/遗漏参数会让"可训练白名单"失真。
        optimizer = torch.optim.AdamW([list(model.parameters())[0]], lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        with self.assertRaisesRegex(ValueError, "whitelist"):
            run_accumulation(
                optimizer=optimizer, trainable=list(model.parameters()), micro_batches=[0, 1],
                forward_loss=factory, accumulation=2, device=torch.device("cpu"),
                autocast=autocast, scaler=scaler,
            )

    def test_empty_trainable_rejected(self):
        model, batches = self._setup()
        factory = make_forward_loss(model, batches)
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        with self.assertRaises(ValueError):
            run_accumulation(
                optimizer=optimizer, trainable=(), micro_batches=[0, 1],
                forward_loss=factory, accumulation=2, device=torch.device("cpu"),
                autocast=autocast, scaler=scaler,
            )

    def test_non_scalar_loss_rejected(self):
        model, batches = self._setup()
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        with self.assertRaisesRegex(ValueError, "scalar"):
            run_accumulation(
                optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
                forward_loss=lambda _micro: torch.zeros(2, requires_grad=True),
                accumulation=2, device=torch.device("cpu"), autocast=autocast, scaler=scaler,
            )

    def test_nonfinite_loss_rejected(self):
        model, batches = self._setup()
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        with self.assertRaisesRegex(ValueError, "Nonfinite training loss"):
            run_accumulation(
                optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
                forward_loss=lambda _micro: torch.tensor(float("nan"), requires_grad=True),
                accumulation=2, device=torch.device("cpu"), autocast=autocast, scaler=scaler,
            )

    def test_all_zero_gradient_rejected(self):
        """全零梯度说明没有真正学习，必须报错而不是当成成功的一步。

        loss 与模型参数相连但梯度恒为零（乘以 0），因此走的是"梯度存在但全为零"这条
        分支，而不是"梯度缺失"那条；两者都必须拒绝。
        """
        model, batches = self._setup()
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            run_accumulation(
                optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
                forward_loss=lambda _micro: (model(torch.zeros(1, 3)) * 0.0).sum(),
                accumulation=2, device=torch.device("cpu"), autocast=autocast, scaler=scaler,
            )

    def test_gradients_cleared_after_success_and_failure(self):
        """正常与异常路径都必须清梯度；不回滚已完成的步进。"""
        model, batches = self._setup()
        factory = make_forward_loss(model, batches)
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        run_accumulation(
            optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
            forward_loss=factory, accumulation=2, device=torch.device("cpu"),
            autocast=autocast, scaler=scaler,
        )
        self.assertTrue(all(parameter.grad is None for parameter in trainable))

        optimizer2 = torch.optim.AdamW(trainable, lr=1e-4)
        with self.assertRaises(ValueError):
            run_accumulation(
                optimizer=optimizer2, trainable=trainable, micro_batches=[0, 1],
                forward_loss=lambda _micro: torch.tensor(float("inf"), requires_grad=True),
                accumulation=2, device=torch.device("cpu"), autocast=autocast, scaler=scaler,
            )
        self.assertTrue(all(parameter.grad is None for parameter in trainable))

    def test_inference_mode_rejected(self):
        model, batches = self._setup()
        factory = make_forward_loss(model, batches)
        trainable = list(model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        autocast, scaler = resolve_amp(torch.device("cpu"), "float16")
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "enabled autograd"):
            run_accumulation(
                optimizer=optimizer, trainable=trainable, micro_batches=[0, 1],
                forward_loss=factory, accumulation=2, device=torch.device("cpu"),
                autocast=autocast, scaler=scaler,
            )


if __name__ == "__main__":
    unittest.main()
