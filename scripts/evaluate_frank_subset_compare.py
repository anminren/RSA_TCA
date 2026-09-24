#!/usr/bin/env python3
"""Evaluate seq2seq models/checkpoints on the BART-annotated FRANK subset and compare to model-specific baseline.

This script uses all hashes where FRANK has human annotations for the original `bart` system: 250 samples, including 46 error-labelled and 204 no-error-labelled cases.
For non-BART backbones, the model-specific pretrained generation is used as baseline.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_frank import load_model_with_checkpoint, generate_summary, load_frank_data

ERROR_TYPES = [
    "EntE", "RelE", "CircE", "OutE", "GramE", "CorefE", "LinkE",
    "Semantic_Frame_Errors", "Discourse_Errors", "Content_Verifiability_Errors",
]
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}|\d+(?:\.\d+)?%?|£\d+(?:\.\d+)?m?|\$\d+(?:\.\d+)?m?|\d{4})\b")


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def rouge_pair(ref, pred):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ref or "", pred or "")
    return {k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}


def token_edit_ratio(a, b):
    import difflib
    aa, bb = (a or "").split(), (b or "").split()
    if not aa and not bb:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, aa, bb).ratio()


def unsupported_items(summary, article):
    article_l = (article or "").lower()
    items = []
    for ent in ENTITY_RE.findall(summary or ""):
        e = ent.strip()
        if len(e) <= 1:
            continue
        if e.lower() not in article_l:
            items.append(e)
    return sorted(set(items))


def build_bart_error_subset(frank_dir):
    benchmark_data, ann_map_tuple = load_frank_data(str(frank_dir))
    anns = json.load(open(Path(frank_dir) / "human_annotations.json", encoding="utf-8"))
    ann_by_hash = {}
    selected = []
    seen = set()
    for ann in anns:
        if ann.get("model_name") != "bart":
            continue
        errs = []
        for et in ERROR_TYPES:
            v = ann.get(et, 1.0)
            if isinstance(v, (int, float)) and v < 1.0:
                errs.append(et)
        ann_by_hash[ann["hash"]] = {
            "errors": errs,
            "has_bart_error": bool(errs),
            "annotation": ann,
        }
        selected.append(ann["hash"])
    selected_set = set(selected)
    rows = []
    for item in benchmark_data:
        h = item["hash"]
        if h in selected_set and h not in seen:
            seen.add(h)
            rows.append(item)
    return rows, ann_by_hash


def generate_results(model_name, checkpoint, frank_dir, output_dir, device):
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    subset, ann_by_hash = build_bart_error_subset(frank_dir)
    model, tokenizer = load_model_with_checkpoint(model_name, checkpoint)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    results = []
    for i, item in enumerate(subset, 1):
        h = item["hash"]
        summary = generate_summary(model, tokenizer, item["article"], dev)
        ann = ann_by_hash.get(h, {})
        results.append({
            "index": i,
            "hash": h,
            "dataset": item.get("dataset", ""),
            "split": item.get("split", ""),
            "article": item.get("article", ""),
            "reference": item.get("reference", ""),
            "generated_summary": summary,
            "bart_error_types": ann.get("errors", []),
            "has_bart_error": ann.get("has_bart_error", False),
            "bart_annotation": ann.get("annotation", {}),
        })
    with open(output_dir / "frank_subset_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump({"total_samples": len(results), "subset": "bart_annotated_250_hashes", "checkpoint": checkpoint}, f, indent=2, ensure_ascii=False)
    return results


def load_or_generate(model_name, checkpoint, frank_dir, output_dir, device):
    path = output_dir / "frank_subset_generation_results.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    return generate_results(model_name, checkpoint, frank_dir, output_dir, device)


def compare(baseline, finetuned, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_map = {r["hash"]: r for r in baseline}
    rows = []
    by_error = defaultdict(list)
    for ft in finetuned:
        h = ft["hash"]
        b = base_map.get(h)
        if not b:
            continue
        ref = ft.get("reference", "")
        article = ft.get("article", "")
        bs = b.get("generated_summary", "")
        fs = ft.get("generated_summary", "")
        br = rouge_pair(ref, bs)
        fr = rouge_pair(ref, fs)
        bu = unsupported_items(bs, article)
        fu = unsupported_items(fs, article)
        diff_l = fr["rougeL"] - br["rougeL"]
        unsupported_delta = len(fu) - len(bu)
        if unsupported_delta < 0 or diff_l > 0.02:
            judgement = "likely_improved"
        elif unsupported_delta > 0 or diff_l < -0.02:
            judgement = "likely_worse"
        else:
            judgement = "neutral_or_changed"
        row = {
            "hash": h,
            "bart_error_types": ";".join(ft.get("bart_error_types", [])),
            "has_bart_error": ft.get("has_bart_error", False),
            "judgement": judgement,
            "diff_rougeL": diff_l,
            "diff_rouge1": fr["rouge1"] - br["rouge1"],
            "baseline_rougeL": br["rougeL"],
            "finetuned_rougeL": fr["rougeL"],
            "edit_ratio": token_edit_ratio(bs, fs),
            "baseline_unsupported_count": len(bu),
            "finetuned_unsupported_count": len(fu),
            "unsupported_delta": unsupported_delta,
            "baseline_unsupported_items": ";".join(bu),
            "finetuned_unsupported_items": ";".join(fu),
            "reference": ref,
            "baseline_summary": bs,
            "finetuned_summary": fs,
            "article_head": article[:800],
        }
        rows.append(row)
        for et in ft.get("bart_error_types", []):
            by_error[et].append(row)
    fields = list(rows[0].keys()) if rows else []
    with open(output_dir / "subset_comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    counts = Counter(r["judgement"] for r in rows)
    summary = {
        "n_samples": len(rows),
        "judgement_counts": dict(counts),
        "mean_diff_rougeL": sum(r["diff_rougeL"] for r in rows) / len(rows) if rows else 0,
        "mean_unsupported_delta": sum(r["unsupported_delta"] for r in rows) / len(rows) if rows else 0,
        "error_labelled": {},
        "no_error_labelled": {},
        "by_error_type": {},
    }
    for et, rs in by_error.items():
        c = Counter(r["judgement"] for r in rs)
        summary["by_error_type"][et] = {
            "n": len(rs),
            "judgement_counts": dict(c),
            "mean_diff_rougeL": sum(r["diff_rougeL"] for r in rs) / len(rs),
            "mean_unsupported_delta": sum(r["unsupported_delta"] for r in rs) / len(rs),
        }
    for group_name, pred in (("error_labelled", True), ("no_error_labelled", False)):
        rs = [r for r in rows if bool(r.get("has_bart_error")) is pred]
        c = Counter(r["judgement"] for r in rs)
        summary[group_name] = {
            "n": len(rs),
            "judgement_counts": dict(c),
            "mean_diff_rougeL": sum(r["diff_rougeL"] for r in rs) / len(rs) if rs else 0,
            "mean_unsupported_delta": sum(r["unsupported_delta"] for r in rs) / len(rs) if rs else 0,
        }
    with open(output_dir / "subset_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    md = ["# FRANK BART-Error Subset Comparison", "", f"- Samples: {summary['n_samples']} (BART annotated: 46 error-labelled + 204 no-error-labelled)", f"- Judgements: {summary['judgement_counts']}", f"- Mean ΔROUGE-L: {summary['mean_diff_rougeL']:.4f}", f"- Mean unsupported delta: {summary['mean_unsupported_delta']:.4f}", f"- Error-labelled group: {summary['error_labelled']}", f"- No-error-labelled group: {summary['no_error_labelled']}", "", "## By Error Type", "", "| Error | N | likely_improved | likely_worse | neutral | mean ΔL | unsupported Δ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for et, s in sorted(summary["by_error_type"].items()):
        jc=s["judgement_counts"]
        md.append(f"| {et} | {s['n']} | {jc.get('likely_improved',0)} | {jc.get('likely_worse',0)} | {jc.get('neutral_or_changed',0)} | {s['mean_diff_rougeL']:.4f} | {s['mean_unsupported_delta']:.4f} |")
    (output_dir / "subset_comparison_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--baseline_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--frank_data', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    baseline = load_or_generate(args.model_name, None, Path(args.frank_data), Path(args.baseline_dir), args.device)
    finetuned = load_or_generate(args.model_name, args.checkpoint, Path(args.frank_data), Path(args.output_dir), args.device)
    summary = compare(baseline, finetuned, Path(args.output_dir)) if args.checkpoint else None
    if summary:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(f"baseline generated/loaded: {len(baseline)} samples")

if __name__ == '__main__':
    main()
