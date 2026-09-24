"""按 FRANK 错误类型分层分析：在原始 BART 有错误的样本上，EEG 微调是否改善。

原理：
1. 读取 human_annotations.json，找到原始 BART 在特定错误类型上得分 < 1.0 的 hash
2. 在这些 hash 的子集上，对比 EEG 微调 vs 原始 BART 的 ROUGE
3. 同时统计摘要文本的编辑距离/变化率

用法:
    python scripts/analyze_frank_by_errors.py \
        --finetuned outputs/frank_eval/bart_derco_simple_raw/ACB71_best_fold1/frank_generation_results.json \
        --baseline outputs/frank_eval/baseline/frank_generation_results.json \
        --frank_data /path/to/frank/data \
        --output outputs/frank_eval/bart_derco_simple_raw/ACB71_best_fold1/error_analysis.json

批量运行:
    python scripts/analyze_frank_by_errors.py --batch
"""

import argparse
import json
import sys
from pathlib import Path

from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, pred in zip(references, predictions):
        if not ref or not pred:
            continue
        scores = scorer.score(ref, pred)
        for key in results:
            results[key].append(scores[key].fmeasure)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in results.items()}


def token_edit_distance(a: str, b: str):
    """简单的 token 级编辑距离比例。"""
    ta = a.split()
    tb = b.split()
    import difflib
    sm = difflib.SequenceMatcher(None, ta, tb)
    return 1 - sm.ratio()


def load_annotations(frank_data_dir: Path):
    with open(frank_data_dir / "human_annotations.json", "r", encoding="utf-8") as f:
        anns = json.load(f)
    # 只取 bart 模型的标注
    map_by_hash = {}
    for ann in anns:
        if ann.get("model_name") == "bart":
            map_by_hash[ann["hash"]] = ann
    return map_by_hash


def analyze_one(finetuned_path: Path, baseline_path: Path, ann_map: dict, error_types: list):
    ft_data = json.load(open(finetuned_path, "r", encoding="utf-8"))
    base_data = json.load(open(baseline_path, "r", encoding="utf-8"))
    base_map = {r["hash"]: r for r in base_data}

    # 按错误类型分组 hash
    error_hashes = {et: set() for et in error_types}
    for h, ann in ann_map.items():
        for et in error_types:
            score = ann.get(et, 1.0)
            if score is not None and score < 1.0:
                error_hashes[et].add(h)

    # 全局对比
    all_valid = []
    for r in ft_data:
        h = r["hash"]
        ref = r.get("reference", "")
        ft = r.get("finetuned_summary", "")
        orig = base_map.get(h, {}).get("finetuned_summary", "")
        if ref and ft and orig:
            all_valid.append({"hash": h, "ref": ref, "ft": ft, "orig": orig})

    global_result = _compare_subset(all_valid)

    # 各错误类型子集对比
    error_results = {}
    for et in error_types:
        hashes = error_hashes[et]
        subset = [x for x in all_valid if x["hash"] in hashes]
        if subset:
            error_results[et] = _compare_subset(subset)
            error_results[et]["num_samples"] = len(subset)

    return {
        "name": finetuned_path.parent.name,
        "global": global_result,
        "by_error_type": error_results,
    }


def _compare_subset(samples):
    refs = [s["ref"] for s in samples]
    fts = [s["ft"] for s in samples]
    origs = [s["orig"] for s in samples]

    ft_rouge = compute_rouge(refs, fts)
    orig_rouge = compute_rouge(refs, origs)

    edits = [token_edit_distance(o, f) for o, f in zip(origs, fts)]
    avg_edit = sum(edits) / len(edits) if edits else 0.0

    return {
        "num_samples": len(samples),
        "finetuned_vs_ref": ft_rouge,
        "original_vs_ref": orig_rouge,
        "diff_rouge1": ft_rouge["rouge1"] - orig_rouge["rouge1"],
        "diff_rouge2": ft_rouge["rouge2"] - orig_rouge["rouge2"],
        "diff_rougeL": ft_rouge["rougeL"] - orig_rouge["rougeL"],
        "avg_token_edit_ratio": avg_edit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", help="微调后的 frank_generation_results.json")
    parser.add_argument("--baseline", default="outputs/frank_eval/baseline/frank_generation_results.json")
    parser.add_argument("--frank_data", required=True)
    parser.add_argument("--output", help="输出 JSON 路径")
    parser.add_argument("--batch", action="store_true", help="批量处理所有被试")
    args = parser.parse_args()

    frank_data_dir = Path(args.frank_data)
    ann_map = load_annotations(frank_data_dir)

    error_types = [
        "EntE", "RelE", "CircE", "OutE",
        "Semantic_Frame_Errors", "Discourse_Errors", "Content_Verifiability_Errors"
    ]

    if args.batch:
        base_dir = Path("outputs/frank_eval")
        all_results = sorted(base_dir.rglob("frank_generation_results.json"))
        result_paths = [p for p in all_results if "baseline" not in str(p)]

        rows = []
        for fp in result_paths:
            name = fp.parent.name
            print(f"[ErrorAnalysis] {name} ...", end=" ")
            out_path = fp.parent / "error_analysis.json"
            result = analyze_one(fp, Path(args.baseline), ann_map, error_types)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            rows.append(result)
            print("OK")

        # 汇总表
        print("\n" + "=" * 100)
        print("错误类型分层 ROUGE 对比（在原始 BART 有错误的样本子集上）")
        print("=" * 100)

        for et in error_types:
            print(f"\n### 错误类型: {et}")
            header = f"{'Subject/Fold':<25} {'N':>6} {'FT-R1':>8} {'Base-R1':>8} {'Diff-R1':>8} {'Edit%':>8}"
            print(header)
            print("-" * len(header))
            diffs = []
            for r in rows:
                if et in r["by_error_type"]:
                    d = r["by_error_type"][et]
                    print(f"{r['name']:<25} {d['num_samples']:>6} "
                          f"{d['finetuned_vs_ref']['rouge1']:>8.4f} {d['original_vs_ref']['rouge1']:>8.4f} "
                          f"{d['diff_rouge1']:>+8.4f} {d['avg_token_edit_ratio']*100:>7.1f}%")
                    diffs.append(d["diff_rouge1"])
            if diffs:
                avg = sum(diffs) / len(diffs)
                improved = sum(1 for d in diffs if d > 0)
                print(f"  Average diff: {avg:+.4f}, Improved: {improved}/{len(diffs)} ({improved/len(diffs)*100:.1f}%)")

        # 保存总汇总
        summary_path = base_dir / "error_analysis_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\n总汇总已保存: {summary_path}")

    else:
        result = analyze_one(Path(args.finetuned), Path(args.baseline), ann_map, error_types)
        out = Path(args.output) if args.output else Path(args.finetuned).parent / "error_analysis.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"分析结果已保存: {out}")


if __name__ == "__main__":
    main()
