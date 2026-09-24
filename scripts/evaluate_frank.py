"""FRANK 数据集摘要事实性评估脚本。

本脚本用于评估 EEG 微调后的 BART 在 FRANK benchmark 上的摘要质量，
重点不是追求高分，而是观察特定类型事实性错误的改善情况。

用法:
    # 使用原始 BART-large-cnn 生成基线摘要
    python scripts/evaluate_frank.py \
        --model_name facebook/bart-large-cnn \
        --frank_data /path/to/frank/data \
        --output_dir outputs/frank_eval/baseline

    # 使用 EEG 微调后的 checkpoint 生成摘要
    python scripts/evaluate_frank.py \
        --model_name facebook/bart-large-cnn \
        --checkpoint outputs/bart_derco_simple/ACB71/best_model.pt \
        --frank_data /path/to/frank/data \
        --output_dir outputs/frank_eval/finetuned

    # 只生成 FRANK 中 BART 原本有特定错误的样本（用于重点分析）
    python scripts/evaluate_frank.py \
        --model_name facebook/bart-large-cnn \
        --frank_data /path/to/frank/data \
        --output_dir outputs/frank_eval/analysis \
        --filter_errors EntE OutE RelE \
        --max_samples 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = TaskType = get_peft_model = None

logger = logging.getLogger("eeg_bart.frank_eval")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_model_with_checkpoint(model_name: str, checkpoint_path: str = None):
    """加载 seq2seq 模型，可选加载 EEG 微调后的 encoder checkpoint。"""
    is_local = Path(model_name).is_dir()
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=is_local)
    use_fast = not getattr(cfg, "model_type", "").startswith("pegasus")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        local_files_only=is_local,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=is_local,
        use_fast=use_fast,
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"加载微调 checkpoint: {checkpoint_path}")
        # ``weights_only=True`` prevents checkpoint pickle payloads from
        # importing or executing arbitrary Python objects.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        is_lora = any("lora_A" in k or "lora_B" in k for k in state_dict)

        if is_lora:
            if get_peft_model is None:
                raise ImportError("peft is required to load LoRA checkpoints")
            r = next((v.shape[0] for k, v in state_dict.items() if "lora_A" in k), 8)
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=True,
                r=int(r),
                lora_alpha=int(ckpt.get("lora_alpha", 2 * int(r))),
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            logger.info(f"检测到 LoRA checkpoint，已包装 Seq2SeqLM: r={r}, alpha={2 * int(r)}")

        model_state = model.state_dict()
        loaded = 0
        skipped = 0
        for k, v in state_dict.items():
            candidates = []
            if is_lora:
                if k.startswith("encoder.model.base_model.model."):
                    candidates.append("base_model.model.model." + k[len("encoder.model.base_model.model."):])
                if k.startswith("encoder.model."):
                    candidates.append("base_model.model." + k[len("encoder.model."):])
            if k.startswith("encoder.model."):
                candidates.append("model." + k[len("encoder.model."):])
            elif k.startswith("encoder."):
                candidates.append("model." + k[len("encoder."):])
            candidates.append(k)

            new_k = next((c for c in candidates if c in model_state and model_state[c].shape == v.shape), None)
            if new_k is not None:
                model_state[new_k].copy_(v)
                loaded += 1
            else:
                skipped += 1
                if not k.startswith("brain_head"):
                    logger.warning(f"Checkpoint key {k} 无匹配，跳过")
        logger.info(f"Checkpoint 加载完成: loaded={loaded}, skipped={skipped}, total={len(state_dict)}")
    else:
        if checkpoint_path:
            logger.warning(f"Checkpoint 不存在: {checkpoint_path}，使用原始模型")

    return model, tokenizer


def generate_summary(model, tokenizer, article: str, device, max_length=1024, max_summary_len=142):
    """为单篇文章生成摘要。"""
    model.eval()
    inputs = tokenizer(
        article,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_summary_len,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def load_frank_data(frank_data_dir: str):
    """加载 FRANK benchmark 数据和人类标注。"""
    frank_dir = Path(frank_data_dir)

    with open(frank_dir / "benchmark_data.json", "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    with open(frank_dir / "human_annotations.json", "r", encoding="utf-8") as f:
        human_annotations = json.load(f)

    # 将 human annotations 按 (hash, model_name) 索引
    ann_map = {}
    for ann in human_annotations:
        key = (ann["hash"], ann["model_name"])
        ann_map[key] = ann

    return benchmark_data, ann_map


def filter_by_errors(benchmark_data, ann_map, error_types, target_model="bart"):
    """筛选出目标模型在指定错误类型上有缺陷的样本（按 hash 去重）。"""
    seen_hashes = set()
    filtered = []
    for item in benchmark_data:
        hash_val = item["hash"]
        if hash_val in seen_hashes:
            continue
        key = (hash_val, target_model)
        if key not in ann_map:
            continue
        ann = ann_map[key]
        has_error = False
        for err_type in error_types:
            if err_type in ann and ann[err_type] < 1.0:
                has_error = True
                break
        if has_error:
            seen_hashes.add(hash_val)
            filtered.append(item)
    return filtered


def run_generation(model, tokenizer, benchmark_data, ann_map, device, args):
    """运行摘要生成并收集结果。"""
    results = []
    target_model = "bart"

    # 预先构建 hash -> 原始 BART 摘要 的映射
    bart_summary_map = {}
    for bd in benchmark_data:
        if bd["model_name"] == target_model:
            bart_summary_map[bd["hash"]] = bd["summary"]

    seen_hashes = set()
    for item in tqdm(benchmark_data, desc="生成摘要"):
        hash_val = item["hash"]
        if hash_val in seen_hashes:
            continue
        seen_hashes.add(hash_val)

        article = item["article"]
        reference = item.get("reference", "")

        # 生成微调后的摘要
        generated = generate_summary(
            model, tokenizer, article, device,
            max_length=args.max_article_length,
            max_summary_len=args.max_summary_length,
        )

        # 获取 FRANK 中原始 BART 的摘要和人类标注
        orig_bart_summary = bart_summary_map.get(hash_val, "")
        annotations = {}
        key = (hash_val, target_model)
        if key in ann_map:
            ann = ann_map[key]
            annotations = {
                "Factuality": ann.get("Factuality", None),
                "Semantic_Frame_Errors": ann.get("Semantic_Frame_Errors", None),
                "Discourse_Errors": ann.get("Discourse_Errors", None),
                "Content_Verifiability_Errors": ann.get("Content_Verifiability_Errors", None),
                "RelE": ann.get("RelE", None),
                "EntE": ann.get("EntE", None),
                "CircE": ann.get("CircE", None),
                "OutE": ann.get("OutE", None),
                "GramE": ann.get("GramE", None),
                "CorefE": ann.get("CorefE", None),
                "LinkE": ann.get("LinkE", None),
            }

        results.append({
            "hash": hash_val,
            "dataset": item.get("dataset", "unknown"),
            "split": item.get("split", "unknown"),
            "model_name": target_model,
            "article": article,
            "reference": reference,
            "original_bart_summary": orig_bart_summary,
            "finetuned_summary": generated,
            "annotations": annotations,
        })

    return results


def compute_simple_metrics(results):
    """计算简单的 token 重叠指标（不依赖外部包）。"""
    # 由于用户说"不一定需要量化指标"，这里只做最基础的统计
    stats = {
        "total_samples": len(results),
        "avg_article_length": 0,
        "avg_original_summary_length": 0,
        "avg_finetuned_summary_length": 0,
    }

    article_lens = []
    orig_lens = []
    ft_lens = []

    for r in results:
        article_lens.append(len(r["article"].split()))
        orig_lens.append(len(r["original_bart_summary"].split()))
        ft_lens.append(len(r["finetuned_summary"].split()))

    if article_lens:
        stats["avg_article_length"] = sum(article_lens) / len(article_lens)
        stats["avg_original_summary_length"] = sum(orig_lens) / len(orig_lens)
        stats["avg_finetuned_summary_length"] = sum(ft_lens) / len(ft_lens)

    return stats


def analyze_error_cases(results, error_types):
    """分析在特定错误类型上，微调前后摘要的变化（按 hash 去重）。"""
    case_studies = []
    seen_hashes = set()

    for r in results:
        hash_val = r["hash"]
        if hash_val in seen_hashes:
            continue
        ann = r["annotations"]
        for err_type in error_types:
            score = ann.get(err_type, 1.0)
            if score is not None and score < 1.0:
                seen_hashes.add(hash_val)
                case_studies.append({
                    "hash": hash_val,
                    "error_type": err_type,
                    "error_score": score,
                    "factuality": ann.get("Factuality", None),
                    "article": r["article"][:500],
                    "reference": r["reference"][:300],
                    "original_bart_summary": r["original_bart_summary"],
                    "finetuned_summary": r["finetuned_summary"],
                })
                break  # 一个样本只记录一次

    return case_studies


def generate_report(results, stats, case_studies, output_dir: Path):
    """生成对比分析报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    with open(output_dir / "frank_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存统计
    with open(output_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 保存案例分析
    with open(output_dir / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2, ensure_ascii=False)

    # 生成 Markdown 报告
    md_lines = [
        "# FRANK 摘要事实性评估报告",
        "",
        "## 1. 基本信息",
        "",
        f"- 总样本数: {stats['total_samples']}",
        f"- 平均文章长度: {stats['avg_article_length']:.1f} tokens",
        f"- 平均原始 BART 摘要长度: {stats['avg_original_summary_length']:.1f} tokens",
        f"- 平均微调后摘要长度: {stats['avg_finetuned_summary_length']:.1f} tokens",
        "",
        "## 2. 重点错误案例分析",
        "",
        f"共筛选出 {len(case_studies)} 个原始 BART 存在事实性错误的样本。",
        "",
    ]

    # 按错误类型分组
    from collections import defaultdict
    cases_by_error = defaultdict(list)
    for case in case_studies:
        cases_by_error[case["error_type"]].append(case)

    for err_type, cases in sorted(cases_by_error.items()):
        md_lines.append(f"### {err_type} ({len(cases)} 例)")
        md_lines.append("")
        for i, case in enumerate(cases[:10], 1):  # 每类最多展示 10 例
            md_lines.append(f"#### 案例 {i} (hash: {case['hash']})")
            md_lines.append(f"- 事实性得分: {case['factuality']}")
            md_lines.append(f"- 错误得分: {case['error_score']}")
            md_lines.append("")
            md_lines.append("**原文 (前 300 字):**")
            md_lines.append(f"> {case['article'][:300]}...")
            md_lines.append("")
            md_lines.append("**Reference:**")
            md_lines.append(f"> {case['reference']}")
            md_lines.append("")
            md_lines.append("**原始 BART 摘要:**")
            md_lines.append(f"> {case['original_bart_summary']}")
            md_lines.append("")
            md_lines.append("**EEG 微调后摘要:**")
            md_lines.append(f"> {case['finetuned_summary']}")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

    md_lines.append("## 3. 观察结论模板")
    md_lines.append("")
    md_lines.append("请人工审阅以上案例，填写以下观察：")
    md_lines.append("")
    md_lines.append("- **实体错误 (EntE) 改善情况**: ___")
    md_lines.append("- **关系错误 (RelE) 改善情况**: ___")
    md_lines.append("- **文章外错误 (OutE) 改善情况**: ___")
    md_lines.append("- **情境错误 (CircE) 改善情况**: ___")
    md_lines.append("- **语篇错误 (Discourse) 改善情况**: ___")
    md_lines.append("- **语义框架错误 (Semantic Frame) 改善情况**: ___")
    md_lines.append("- **意外退化案例**: ___")
    md_lines.append("")

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"报告已保存至: {output_dir}")
    logger.info(f"  - 完整结果: frank_generation_results.json")
    logger.info(f"  - 案例分析: case_studies.json")
    logger.info(f"  - 阅读报告: report.md")


def main():
    parser = argparse.ArgumentParser(description="FRANK 摘要事实性评估")
    parser.add_argument("--model_name", required=True, help="Hugging Face 模型名或本地模型路径")
    parser.add_argument("--checkpoint", default=None, help="EEG 微调后的 checkpoint 路径")
    parser.add_argument("--frank_data", required=True, help="FRANK 数据目录")
    parser.add_argument("--output_dir", default="outputs/frank_eval", help="输出目录")
    parser.add_argument("--filter_errors", nargs="+", default=None,
                        help="只生成原始 BART 在指定错误类型上有缺陷的样本，如: EntE OutE RelE")
    parser.add_argument("--max_samples", type=int, default=None, help="最大生成样本数")
    parser.add_argument("--max_article_length", type=int, default=1024, help="文章最大长度")
    parser.add_argument("--max_summary_length", type=int, default=142, help="摘要最大长度")
    parser.add_argument("--device", default="cuda", help="计算设备")
    args = parser.parse_args()

    setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载模型
    model, tokenizer = load_model_with_checkpoint(args.model_name, args.checkpoint)
    model.to(device)
    model.eval()

    # 加载 FRANK 数据
    logger.info(f"加载 FRANK 数据: {args.frank_data}")
    benchmark_data, ann_map = load_frank_data(args.frank_data)
    logger.info(f"FRANK 总样本数: {len(benchmark_data)}")

    # 筛选数据
    if args.filter_errors:
        benchmark_data = filter_by_errors(benchmark_data, ann_map, args.filter_errors, target_model="bart")
        logger.info(f"筛选后样本数 (原始 BART 有 {args.filter_errors} 错误): {len(benchmark_data)}")

    if args.max_samples:
        benchmark_data = benchmark_data[:args.max_samples]
        logger.info(f"限制为前 {args.max_samples} 个样本")

    # 生成摘要
    logger.info("开始生成摘要...")
    results = run_generation(model, tokenizer, benchmark_data, ann_map, device, args)

    # 计算基础统计
    stats = compute_simple_metrics(results)
    logger.info(f"平均文章长度: {stats['avg_article_length']:.1f}")
    logger.info(f"平均原始摘要长度: {stats['avg_original_summary_length']:.1f}")
    logger.info(f"平均微调后摘要长度: {stats['avg_finetuned_summary_length']:.1f}")

    # 错误案例分析
    error_types = args.filter_errors or ["EntE", "OutE", "RelE", "CircE", "Semantic_Frame_Errors", "Discourse_Errors", "Content_Verifiability_Errors"]
    case_studies = analyze_error_cases(results, error_types)
    logger.info(f"重点错误案例数: {len(case_studies)}")

    # 生成报告
    output_dir = Path(args.output_dir)
    generate_report(results, stats, case_studies, output_dir)


if __name__ == "__main__":
    main()
