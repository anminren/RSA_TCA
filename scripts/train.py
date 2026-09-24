"""BART EEG 微调主训练入口。

支持两种实现模式：
- simple: BartEncoder + Linear Head + MSE (类似 ECoG-Tuning-main)
- moco: Brain-MoCo 完整框架 (类似 EEG-Tuning-Llama)

支持两种数据集：
- derco: Derco 数据集 (32ch, 1000Hz, 22 subjects)
- frank: Frank 数据集 (28ch, 250Hz, 13 subjects)

用法示例:
    # Derco simple
    python scripts/train.py --config configs/train/bart_derco_simple.yaml --subject ACB71

    # Frank simple
    python scripts/train.py --config configs/train/bart_frank_simple.yaml --frank_subject sub_1

    # Derco MoCo
    python scripts/train.py --config configs/train/bart_derco_moco.yaml --subject ACB71

    # 切换模型为 large-cnn
    python scripts/train.py --config configs/train/bart_derco_simple.yaml --subject ACB71 \
        --set model.model_name=facebook/bart-large-cnn

    # K折交叉验证
    python scripts/train.py --config configs/train/bart_derco_simple.yaml --subject ACB71 --n_folds 5 --fold 0
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import (
    StimulusLoader,
    FrankStimulusLoader,
    TextEEGDataset,
    WordCategoryCache,
    eeg_collate_fn,
)
from models import BartBackbone, BartSimple, BartBrainMoCo
from losses import CombinedLoss
from training import EEGBARTTrainer
from utils import load_config, set_seed

logger = logging.getLogger("eeg_bart")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_model(config):
    """根据配置构建模型。"""
    mcfg = config.model

    if mcfg.get("enable_moco", False):
        model = BartBrainMoCo(
            model_name=mcfg.model_name,
            n_channels=mcfg.n_channels,
            pooling=mcfg.pooling,
            brain_head_dropout=mcfg.brain_head_dropout,
            contrastive_proj_dim=mcfg.get("contrastive_proj_dim", 256),
            ema_momentum=mcfg.get("ema_momentum", 0.999),
            enable_moco=True,
            freeze_encoder=mcfg.get("freeze_encoder", False),
            frozen_anchor=mcfg.get("frozen_anchor", False),
            freeze_decoder=mcfg.get("freeze_decoder", True),
            gradient_checkpointing=mcfg.get("gradient_checkpointing", False),
        )
        logger.info("使用 Brain-MoCo 模型 (enable_moco=True)")
    else:
        model = BartSimple(
            model_name=mcfg.model_name,
            n_channels=mcfg.n_channels,
            pooling=mcfg.pooling,
            brain_head_dropout=mcfg.brain_head_dropout,
            freeze_decoder=mcfg.get("freeze_decoder", True),
            gradient_checkpointing=mcfg.get("gradient_checkpointing", False),
            output_hidden_states=mcfg.get("output_hidden_states", True),
            lora_config=mcfg.get("lora", None),
        )
        if mcfg.get("lora", {}).get("enabled", False):
            logger.info(f"使用简单模型 + LoRA: {mcfg.get('lora')}")
        else:
            logger.info("使用简单模型 (BartEncoder + Linear Head)")

    return model

def _build_time_windows(dcfg):
    """从配置构建时间窗口字典。

    支持两种格式：
    - 旧格式: time_windows: null 或 time_windows: {ner: [150, 350], pos: [0, 500]}
    - Llama格式: time_windows: {enabled: true, ner: {start_ms: 150, end_ms: 350}, pos: {start_ms: 0, end_ms: 500}}
    """
    tw_cfg = dcfg.get("time_windows", None)
    if tw_cfg is None:
        return None

    # 旧格式：直接是字典且没有 enabled 键
    if isinstance(tw_cfg, dict) and "enabled" not in tw_cfg:
        return tw_cfg if tw_cfg else None

    # Llama 格式
    if not tw_cfg.get("enabled", False):
        return None

    time_windows = {}
    if "ner" in tw_cfg:
        time_windows["ner"] = (tw_cfg.ner.start_ms, tw_cfg.ner.end_ms)
    if "pos" in tw_cfg:
        time_windows["pos"] = (tw_cfg.pos.start_ms, tw_cfg.pos.end_ms)
    return time_windows if time_windows else None


def build_datasets(config, tokenizer, args):
    """构建训练和验证数据集。"""
    dcfg = config.data

    if dcfg.dataset_type == "derco":
        if args.subject is None:
            raise ValueError("Derco dataset requires --subject")
        subject_id = args.subject
        stimulus_loader = StimulusLoader(
            data_dir=dcfg.stimulus_dir,
            article_ids=dcfg.article_ids,
        )
    elif dcfg.dataset_type == "frank":
        if args.frank_subject is None:
            raise ValueError("Frank dataset requires --frank_subject")
        subject_id = args.frank_subject
        stimulus_loader = FrankStimulusLoader(
            data_dir=dcfg.stimulus_dir,
            article_ids=dcfg.article_ids,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dcfg.dataset_type}")

    time_windows = _build_time_windows(dcfg)
    word_category_cache = None
    if time_windows is not None:
        word_category_cache = WordCategoryCache()
        logger.info(f"词汇类别时间窗口已启用: {time_windows}")

    common_kwargs = {
        "stimulus_loader": stimulus_loader,
        "tokenizer": tokenizer,
        "subject_id": subject_id,
        "article_ids": dcfg.article_ids,
        "eeg_data_dir": dcfg.eeg_data_dir,
        "max_length": dcfg.max_length,
        "target_type": dcfg.target_type,
        "split_ratio": tuple(dcfg.split_ratio),
        "split_seed": dcfg.split_seed,
        "dataset_type": dcfg.dataset_type,
        "sampling_rate": dcfg.sampling_rate,
        "baseline_ms": dcfg.baseline_ms,
        "max_channels": dcfg.max_channels,
        "time_windows": time_windows,
        "word_category_cache": word_category_cache,
        "max_timepoints": dcfg.get("max_timepoints", None),
    }

    if args.n_folds is not None and args.fold is not None:
        train_dataset = TextEEGDataset(
            **common_kwargs,
            split="train",
            n_folds=args.n_folds,
            current_fold=args.fold,
        )
        val_dataset = TextEEGDataset(
            **common_kwargs,
            split="val",
            n_folds=args.n_folds,
            current_fold=args.fold,
        )
    else:
        train_dataset = TextEEGDataset(**common_kwargs, split="train")
        val_dataset = TextEEGDataset(**common_kwargs, split="val")

    return train_dataset, val_dataset


def main():
    from utils.config import parse_config_args

    args = parse_config_args()
    setup_logging()

    config = load_config(args.config, cli_overrides=args.overrides)
    set_seed(config.experiment.seed)

    logger.info(f"实验名称: {config.experiment.name}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"模型: {config.model.model_name}")
    logger.info(f"数据集类型: {config.data.dataset_type}")

    if args.subject:
        logger.info(f"被试 (Derco): {args.subject}")
    if args.frank_subject:
        logger.info(f"被试 (Frank): {args.frank_subject}")
    if args.n_folds is not None:
        logger.info(f"K折交叉验证: {args.n_folds} folds, fold={args.fold}")

    # === Tokenizer ===
    tokenizer = BartBackbone.load_tokenizer(config.model.model_name)

    # === 数据集 ===
    train_dataset, val_dataset = build_datasets(config, tokenizer, args)
    logger.info(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=eeg_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=eeg_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # === 模型 ===
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,} "
        f"({trainable_params / total_params * 100:.2f}%)"
    )

    # === 损失函数 ===
    lcfg = config.loss
    criterion = CombinedLoss(
        lambda_brain=lcfg.lambda_brain,
        lambda_moco=lcfg.lambda_moco,
        lambda_anchor=lcfg.lambda_anchor,
        moco_temperature=lcfg.get("moco_temperature", 0.2),
        anchor_type=lcfg.anchor_type,
        kl_temperature=lcfg.get("kl_temperature", 0.1),
    )

    # === 训练器 ===
    save_dir = Path("outputs") / config.experiment.name
    if args.subject:
        save_dir = save_dir / args.subject
    elif args.frank_subject:
        save_dir = save_dir / args.frank_subject
    if args.fold is not None:
        save_dir = save_dir / f"fold_{args.fold}"

    trainer = EEGBARTTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=str(save_dir),
    )

    result = trainer.train()
    logger.info(f"训练结果: {result}")


if __name__ == "__main__":
    main()
