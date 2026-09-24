"""基于已有 FRANK 评估结果计算 ROUGE / BERTScore 等量化指标。

用法:
    python scripts/compute_frank_metrics.py \
        --input outputs/frank_eval/baseline/frank_generation_results.json \
        --output outputs/frank_eval/baseline/metrics.json

    # 对比两个结果（原始 BART vs EEG 微调）
    python scripts/compute_frank_metrics.py \
        --input outputs/frank_eval/finetuned/frank_generation_results.json \
        --output outputs/frank_eval/finetuned/metrics.json \
        --compare_baseline outputs/frank_eval/baseline/frank_generation_results.json
"""

import argparse
import json
import sys
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        for key in results:
            results[key].append(scores[key].fmeasure)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in results.items()}


def compute_bertscore(references, predictions, device="cuda", model_type="roberta-large"):
    if not predictions:
        return {"bertscore_f1": 0.0}
    try:
        _, _, f1 = bert_score(
            predictions,
            references,
            model_type=model_type,
            lang="en",
            device=device,
            verbose=False,
            batch_size=32,
        )
        return {"bertscore_f1": f1.mean().item()}
    except Exception as e:
        print(f"[警告] BERTScore 计算失败（可能需要下载模型或网络连接）: {e}")
        return {"bertscore_f1": None}


def compute_length_stats(texts):
    lens = [len(t.split()) for t in texts if t]
    if not lens:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    lens_sorted = sorted(lens)
    n = len(lens_sorted)
    return {
        "mean": sum(lens) / n,
        "median": lens_sorted[n // 2] if n % 2 else (lens_sorted[n // 2 - 1] + lens_sorted[n // 2]) / 2,
        "min": lens_sorted[0],
        "max": lens_sorted[-1],
    }


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(results, device="cuda", use_bertscore=False):
    refs = []
    preds = []
    orig_preds = []

    for r in results:
        ref = r.get("reference", "")
        pred = r.get("finetuned_summary", "")
        orig = r.get("original_bart_summary", "")
        if not ref or not pred:
            continue
        refs.append(ref)
        preds.append(pred)
        if orig:
            orig_preds.append(orig)

    metrics = {}

    # 长度统计
    metrics["lengths"] = {
        "reference": compute_length_stats(refs),
        "finetuned": compute_length_stats(preds),
    }
    if orig_preds:
        metrics["lengths"]["original_bart"] = compute_length_stats(orig_preds)

    # ROUGE
    metrics["rouge_finetuned_vs_reference"] = compute_rouge(refs, preds)
    if orig_preds:
        # 如果 orig_preds 数量与 refs/preds 不同（部分缺失），只对齐有效部分
        metrics["rouge_original_vs_reference"] = compute_rouge(refs[:len(orig_preds)], orig_preds)

    # BERTScore（默认关闭，需要网络下载模型）
    if use_bertscore:
        metrics["bertscore_finetuned_vs_reference"] = compute_bertscore(refs, preds, device=device)
        if orig_preds:
            metrics["bertscore_original_vs_reference"] = compute_bertscore(
                refs[:len(orig_preds)], orig_preds, device=device
            )

    metrics["num_samples"] = len(preds)
    metrics["num_samples_with_original"] = len(orig_preds)

    return metrics


def compare_results(finetuned_results, baseline_results, device="cuda", use_bertscore=False):
    """逐样本对比原始 BART 和微调后 BART 的指标变化。"""
    # 建立 hash -> original_bart_summary 映射（优先用 baseline 里的，更完整）
    baseline_map = {}
    for r in baseline_results:
        h = r["hash"]
        baseline_map[h] = {
            "reference": r.get("reference", ""),
            "original_bart_summary": r.get("original_bart_summary", ""),
            "finetuned_summary": r.get("finetuned_summary", ""),
        }

    refs = []
    preds_finetuned = []
    preds_original = []

    for r in finetuned_results:
        h = r["hash"]
        ref = r.get("reference", "")
        ft = r.get("finetuned_summary", "")
        orig = baseline_map.get(h, {}).get("original_bart_summary", "")
        if not ref or not ft:
            continue
        refs.append(ref)
        preds_finetuned.append(ft)
        preds_original.append(orig if orig else ft)

    comparison = {}
    # 只对齐有原始摘要的样本
    valid_refs = []
    valid_ft = []
    valid_orig = []
    for r, ft, orig in zip(refs, preds_finetuned, preds_original):
        if orig:
            valid_refs.append(r)
            valid_ft.append(ft)
            valid_orig.append(orig)

    if valid_orig:
        comparison["rouge_finetuned_vs_reference"] = compute_rouge(valid_refs, valid_ft)
        comparison["rouge_original_vs_reference"] = compute_rouge(valid_refs, valid_orig)
        if use_bertscore:
            comparison["bertscore_finetuned_vs_reference"] = compute_bertscore(valid_refs, valid_ft, device=device)
            comparison["bertscore_original_vs_reference"] = compute_bertscore(valid_refs, valid_orig, device=device)
        comparison["num_comparison_samples"] = len(valid_refs)
    else:
        comparison["note"] = "缺少原始 BART 摘要，无法进行有效对比"

    return comparison


def generate_report(metrics, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    md_path = output_path.with_suffix(".md")
    lines = [
        "# FRANK 摘要量化指标报告",
        "",
        f"## 样本统计",
        "",
        f"- 总样本数: {metrics.get('num_samples', 'N/A')}",
        f"- 含原始 BART 摘要的样本数: {metrics.get('num_samples_with_original', 'N/A')}",
        "",
        "## 摘要长度统计",
        "",
    ]

    for name, stats in metrics.get("lengths", {}).items():
        lines.append(f"### {name}")
        lines.append(f"- 平均: {stats['mean']:.1f} tokens")
        lines.append(f"- 中位数: {stats['median']:.1f} tokens")
        lines.append(f"- 范围: [{stats['min']}, {stats['max']}]")
        lines.append("")

    lines.extend([
        "## ROUGE 分数",
        "",
        "| 对比 | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "|------|---------|---------|---------|",
    ])
    if "rouge_finetuned_vs_reference" in metrics:
        r = metrics["rouge_finetuned_vs_reference"]
        lines.append(f"| 微调后 vs Reference | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} |")
    if "rouge_original_vs_reference" in metrics:
        r = metrics["rouge_original_vs_reference"]
        lines.append(f"| 原始 BART vs Reference | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} |")
    lines.append("")

    lines.extend([
        "## BERTScore",
        "",
        "| 对比 | F1 |",
        "|------|-----|",
    ])
    if "bertscore_finetuned_vs_reference" in metrics:
        f1 = metrics["bertscore_finetuned_vs_reference"]["bertscore_f1"]
        if f1 is not None:
            lines.append(f"| 微调后 vs Reference | {f1:.4f} |")
        else:
            lines.append("| 微调后 vs Reference | N/A（计算失败） |")
    if "bertscore_original_vs_reference" in metrics:
        f1 = metrics["bertscore_original_vs_reference"]["bertscore_f1"]
        if f1 is not None:
            lines.append(f"| 原始 BART vs Reference | {f1:.4f} |")
        else:
            lines.append("| 原始 BART vs Reference | N/A（计算失败） |")
    lines.append("")

    if "comparison" in metrics:
        lines.extend([
            "## 公平对比（相同样本子集）",
            "",
            f"对比样本数: {metrics['comparison'].get('num_comparison_samples', 'N/A')}",
            "",
            "### ROUGE",
            "| 对比 | ROUGE-1 | ROUGE-2 | ROUGE-L |",
            "|------|---------|---------|---------|",
        ])
        if "rouge_finetuned_vs_reference" in metrics["comparison"]:
            r = metrics["comparison"]["rouge_finetuned_vs_reference"]
            lines.append(f"| 微调后 vs Reference | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} |")
        if "rouge_original_vs_reference" in metrics["comparison"]:
            r = metrics["comparison"]["rouge_original_vs_reference"]
            lines.append(f"| 原始 BART vs Reference | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"指标已保存: {output_path}")
    print(f"报告已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="计算 FRANK 评估结果的量化指标")
    parser.add_argument("--input", required=True, help="frank_generation_results.json 路径")
    parser.add_argument("--output", required=True, help="输出 metrics.json 路径")
    parser.add_argument("--compare_baseline", default=None, help="基线结果 JSON 路径（用于公平对比）")
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument("--use_bertscore", action="store_true", help="启用 BERTScore（需要网络下载 roberta-large）")
    args = parser.parse_args()

    results = load_results(args.input)
    metrics = evaluate(results, device=args.device, use_bertscore=args.use_bertscore)

    if args.compare_baseline:
        baseline = load_results(args.compare_baseline)
        metrics["comparison"] = compare_results(results, baseline, device=args.device, use_bertscore=args.use_bertscore)

    generate_report(metrics, Path(args.output))


if __name__ == "__main__":
    main()
