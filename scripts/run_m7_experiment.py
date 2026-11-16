"""M7正式实验入口：prepare/summarize不使用GPU；calibrate/train/evaluate须显式--execute。

不运行任何测试、N1、pilot或诊断，不下载/安装，不自动启动下一阶段。所有私有输入和输出
留在所选work_dir；失败只记录类型，不重试或覆盖目录。--execute代表实际GPU作业而非预检。
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    """先解析启动意图再导入项目；-I下显式使用本仓库src，不落到旧editable安装。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'calibrate', 'train', 'evaluate', 'summarize'])
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/m7_formal_autodl.json')
    parser.add_argument('--execute', action='store_true', help='显式启动所选正式GPU阶段；prepare不需要')
    parser.add_argument('--method', choices=['full', 'random_k', 'adaptive_k'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--resume', type=Path, help='完整累积组checkpoint；必须另选新output目录')
    parser.add_argument('--training-dir', type=Path, help='评估时使用的完整训练结果目录')
    args = parser.parse_args()
    if args.phase not in ('prepare', 'summarize') and not args.execute:
        parser.error('GPU阶段需显式--execute；当前未启动任何实验')
    if args.phase in ('train', 'evaluate') and args.method is None:
        parser.error('train/evaluate必须指定--method')
    if args.resume is not None and (args.phase != 'train' or args.output is None):
        parser.error('--resume仅用于train且须指定新的--output目录')
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    sys.path.insert(0, str(ROOT / 'src'))
    from adaptive_lora.m7_experiment import load_context, prepare_data, setup_gpu, calibrate, train, evaluate, summarize
    ctx = load_context(args.profile)
    output = args.output or ctx['work'] / {'prepare': 'prepared', 'calibrate': 'calibration', 'summarize': 'comparison',
                                          'train': 'train-' + str(args.method), 'evaluate': 'eval-' + str(args.method)}[args.phase]
    if output.exists() or output.is_symlink():
        parser.error('输出目录已存在；不覆盖，请选择新的--output')
    try:
        if args.phase == 'prepare':
            result = prepare_data(ctx, output)
        elif args.phase == 'summarize':
            result = summarize(ctx, output)
        else:
            setup_gpu()
            if args.phase == 'calibrate':
                result = calibrate(ctx, output, args.prepared_dir)
            elif args.phase == 'train':
                result = train(ctx, args.method, output, args.prepared_dir, args.selection, args.resume)
            else:
                result = evaluate(ctx, args.method, output, args.prepared_dir, args.selection, args.training_dir)
    except BaseException as error:
        # 只在本阶段已创建的新目录记录失败，不创建假的完成标记或输出原始样本文本。
        if output.is_dir() and not (output / 'failure.json').exists():
            from adaptive_lora.artifacts import write_json
            write_json(output / 'failure.json', {'phase': args.phase, 'exception_type': type(error).__name__,
                       'complete': False, 'automatic_retry': False})
        raise
    print(json.dumps({'phase': args.phase, 'status': result['status'], 'output': str(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
