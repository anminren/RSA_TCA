#!/usr/bin/env python3
"""Paired descriptive diagnostics on articles annotated for the original FRANK BART.

Human labels belong to the annotated summary, not an article or a new generation.
Lexical absence and ROUGE are diagnostics, not factuality judgments.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
SCHEMA_VERSION = 2
ERROR_TYPES = [
    "EntE", "RelE", "CircE", "OutE", "GramE", "CorefE", "LinkE",
    "Semantic_Frame_Errors", "Discourse_Errors", "Content_Verifiability_Errors",
]
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}|\d+(?:\.\d+)?%?|£\d+(?:\.\d+)?m?|\$\d+(?:\.\d+)?m?|\d{4})\b")
TRANSITIONS = ("flagged_to_clear", "clear_to_flagged", "flagged_to_flagged", "clear_to_clear")
ALIGNMENT_STATES = ("matched", "different_summary", "missing_original_summary", "missing_annotation")
RESULTS_NAME = "frank_subset_generation_results.json"
MANIFEST_NAME = "generation_manifest.json"
SCREEN_NOTE = (
    "A lexical flag means a regex-extracted item was not found literally in the "
    "article. Clear does not establish factual consistency; ROUGE measures "
    "reference overlap. Neither diagnostic establishes factual-error correction."
)


def normalize(text):
    """Normalize whitespace only; preserve case, punctuation, and word order."""
    return re.sub(r"\s+", " ", (text or "").strip())


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rouge_pair(ref, pred):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {key: scores[key].fmeasure for key in ("rouge1", "rouge2", "rougeL")}


def token_edit_ratio(a, b):
    import difflib
    return 1.0 - difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def lexically_absent_items(summary, article):
    """Literal diagnostic only; absence can be a paraphrase and presence a lie."""
    article_text = normalize(article).casefold()
    items = set()
    for entity in ENTITY_RE.findall(normalize(summary)):
        entity = entity.strip()
        if len(entity) <= 1:
            continue
        pattern = r"(?<!\w)" + re.escape(entity.casefold()) + r"(?!\w)"
        if not re.search(pattern, article_text):
            items.add(entity)
    return sorted(items)


def annotation_error_status(annotation):
    observed = {}
    for key in ERROR_TYPES:
        value = annotation.get(key)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1):
            observed[key] = value
    errors = [key for key, value in observed.items() if value < 1.0]
    # Missing scores must not silently become a no-error label.
    status = True if errors else (False if all(key in observed for key in ERROR_TYPES[:7]) else None)
    return errors, status


def build_bart_error_subset(frank_dir):
    frank_dir = Path(frank_dir)
    benchmark = read_json(frank_dir / "benchmark_data.json")
    annotations = read_json(frank_dir / "human_annotations.json")
    original_rows = {}
    for item in benchmark:
        if item.get("model_name") != "bart":
            continue
        key = item["hash"]
        if key in original_rows and any(original_rows[key].get(field) != item.get(field)
                                       for field in ("article", "reference", "summary")):
            raise ValueError(f"Conflicting original BART records for {key}")
        original_rows[key] = item
    ann_by_hash = {}
    for ann in annotations:
        if ann.get("model_name") != "bart":
            continue
        key = ann["hash"]
        if key in ann_by_hash and ann_by_hash[key]["annotation"] != ann:
            raise ValueError(f"Conflicting BART annotations for {key}")
        if key not in original_rows:
            raise ValueError(f"Missing annotated original BART summary for {key}")
        errors, has_error = annotation_error_status(ann)
        ann_by_hash[key] = {
            "errors": errors, "has_bart_error": has_error, "annotation": ann,
            "original_summary": original_rows[key].get("summary", ""),
        }
    rows = [item for key, item in original_rows.items() if key in ann_by_hash]
    if not rows:
        raise ValueError("No BART-annotated articles found")
    return rows, ann_by_hash


def generation_context(model_name, checkpoint, frank_dir):
    checkpoint_info = None
    if checkpoint:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        checkpoint_info = {"path": str(path), "sha256": file_sha256(path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": str(Path(model_name).resolve()) if Path(model_name).exists() else model_name,
        "checkpoint": checkpoint_info,
        "dataset_sha256": {name: file_sha256(Path(frank_dir) / name)
                           for name in ("benchmark_data.json", "human_annotations.json")},
        "generator_sha256": file_sha256(Path(__file__).with_name("evaluate_frank.py")),
        "decoding": {"max_article_length": 1024, "max_summary_length": 142,
                     "min_length": 10, "num_beams": 4, "length_penalty": 2.0,
                     "do_sample": False, "early_stopping": True},
    }


def generate_results(model_name, checkpoint, frank_dir, output_dir, device, context=None):
    import torch
    from scripts.evaluate_frank import load_model_with_checkpoint, generate_summary
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = context or generation_context(model_name, checkpoint, frank_dir)
    subset, _ = build_bart_error_subset(frank_dir)
    model, tokenizer = load_model_with_checkpoint(model_name, checkpoint)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    results = []
    for index, item in enumerate(subset, 1):
        results.append({
            "index": index, "hash": item["hash"],
            "dataset": item.get("dataset", ""), "split": item.get("split", ""),
            "article": item["article"], "reference": item.get("reference", ""),
            "generated_summary": generate_summary(model, tokenizer, item["article"], dev),
        })
    write_json(output_dir / RESULTS_NAME, results)
    write_json(output_dir / MANIFEST_NAME, {"context": context,
               "results_sha256": file_sha256(output_dir / RESULTS_NAME)})
    write_json(output_dir / "statistics.json", {
        "total_samples": len(results), "subset": "articles_annotated_for_original_bart",
        "model_name": model_name, "checkpoint": checkpoint,
    })
    return results


def load_or_generate(model_name, checkpoint, frank_dir, output_dir, device, regenerate=False):
    output_dir = Path(output_dir)
    path = output_dir / RESULTS_NAME
    context = generation_context(model_name, checkpoint, frank_dir)
    if path.exists() and not regenerate:
        manifest_path = output_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError("Legacy cache has no generation manifest; use --compare_only for "
                             "diagnostics with unknown provenance, or --regenerate")
        manifest = read_json(manifest_path)
        if manifest.get("context") != context or manifest.get("results_sha256") != file_sha256(path):
            raise ValueError("Generation cache does not match this run; choose a new directory "
                             "or use --regenerate")
        return read_json(path)
    return generate_results(model_name, checkpoint, frank_dir, output_dir, device, context)


def read_cached_for_comparison(output_dir):
    path = Path(output_dir) / RESULTS_NAME
    manifest_path = Path(output_dir) / MANIFEST_NAME
    provenance = {"status": "legacy_unverified", "results_sha256": file_sha256(path)}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("results_sha256") != provenance["results_sha256"]:
            raise ValueError(f"Cached results differ from their manifest: {path}")
        provenance.update(status="saved_manifest_content_verified", manifest=manifest,
                          checkpoint_revalidated=False)
    return read_json(path), provenance


def index_results(results, side):
    if not isinstance(results, list) or not results:
        raise ValueError(f"{side}: expected nonempty generation results")
    indexed = {}
    for row in results:
        key = row.get("hash")
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError(f"{side}: missing or duplicate hash {key!r}")
        for field in ("article", "reference", "generated_summary"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"{side}: missing or invalid {field} for {key}")
        if not normalize(row["article"]) or not normalize(row["generated_summary"]):
            raise ValueError(f"{side}: empty article or generated summary for {key}")
        indexed[key] = row
    return indexed


def aligned_label(summary, metadata):
    if not metadata or not metadata.get("annotation"):
        return "unknown", "missing_annotation"
    original = metadata.get("original_summary")
    if not isinstance(original, str) or not normalize(original):
        return "unknown", "missing_original_summary"
    if normalize(summary) != normalize(original):
        return "unknown", "different_summary"
    label = {True: "error", False: "no_error", None: "unknown"}[metadata.get("has_bart_error")]
    return label, "matched"


def compare(baseline, finetuned, output_dir, ann_by_hash=None, provenance=None):
    base_map, tuned_map = index_results(baseline, "baseline"), index_results(finetuned, "finetuned")
    if base_map.keys() != tuned_map.keys():
        raise ValueError("Baseline and finetuned sample hash sets differ; no silent intersection allowed")
    ann_by_hash = ann_by_hash or {}
    rows = []
    alignment_counts = {side: Counter({state: 0 for state in ALIGNMENT_STATES})
                        for side in ("baseline", "finetuned")}
    for key, b in base_map.items():
        ft = tuned_map[key]
        for field in ("article", "reference"):
            if b[field] != ft[field]:
                raise ValueError(f"Mismatched {field} for {key}")
        ref, article = b["reference"], b["article"]
        bs, fs = b["generated_summary"], ft["generated_summary"]
        br, fr = rouge_pair(ref, bs), rouge_pair(ref, fs)
        bu, fu = lexically_absent_items(bs, article), lexically_absent_items(fs, article)
        metadata = ann_by_hash.get(key, {})
        baseline_label, baseline_status = aligned_label(bs, metadata)
        tuned_label, tuned_status = aligned_label(fs, metadata)
        alignment_counts["baseline"][baseline_status] += 1
        alignment_counts["finetuned"][tuned_status] += 1
        transition = ("flagged" if bu else "clear") + "_to_" + ("flagged" if fu else "clear")
        rows.append({
            "hash": key,
            "original_bart_error_types": ";".join(metadata.get("errors", [])),
            "original_bart_has_error": metadata.get("has_bart_error"),
            "original_bart_summary": metadata.get("original_summary", ""),
            "baseline_frank_label": baseline_label, "finetuned_frank_label": tuned_label,
            "baseline_annotation_status": baseline_status, "finetuned_annotation_status": tuned_status,
            "baseline_summary_sha256": text_sha256(bs), "finetuned_summary_sha256": text_sha256(fs),
            "lexical_screen_transition": transition,
            "baseline_lexically_absent_count": len(bu), "finetuned_lexically_absent_count": len(fu),
            "lexical_absence_delta": len(fu) - len(bu),
            "baseline_lexically_absent_items": ";".join(bu), "finetuned_lexically_absent_items": ";".join(fu),
            "diff_rougeL": fr["rougeL"] - br["rougeL"], "diff_rouge1": fr["rouge1"] - br["rouge1"],
            "baseline_rougeL": br["rougeL"], "finetuned_rougeL": fr["rougeL"],
            "edit_ratio": token_edit_ratio(bs, fs),
            "reference": ref, "baseline_summary": bs, "finetuned_summary": fs, "article": article,
        })
    counts = Counter({transition: 0 for transition in TRANSITIONS})
    counts.update(row["lexical_screen_transition"] for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION, "n_samples": len(rows), "screen_note": SCREEN_NOTE,
        "lexical_screen_transition_counts": dict(counts),
        "annotation_alignment_counts": {side: dict(values) for side, values in alignment_counts.items()},
        "mean_diff_rougeL": sum(row["diff_rougeL"] for row in rows) / len(rows),
        "mean_lexical_absence_delta": sum(row["lexical_absence_delta"] for row in rows) / len(rows),
        "generation_provenance": provenance or {"status": "not_provided"},
        "annotation_scope": "Original FRANK BART labels apply only to matching annotated summary text.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "subset_comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "subset_comparison.json", rows)
    write_json(output_dir / "subset_comparison_summary.json", summary)
    md = ["# FRANK paired descriptive diagnostics", "", f"- Samples: {len(rows)}",
          f"- {SCREEN_NOTE}", "- Original annotation matching preserves case and punctuation.",
          f"- Annotation alignment: {summary['annotation_alignment_counts']}",
          f"- Mean change in ROUGE-L: {summary['mean_diff_rougeL']:.4f}",
          f"- Mean lexical absence count change: {summary['mean_lexical_absence_delta']:.4f}",
          "", "| Lexical screen transition | Count |", "|---|---:|"]
    md.extend(f"| {key} | {counts[key]} |" for key in TRANSITIONS)
    md.extend(["", "These counts are not factuality correction or preservation rates.",
               "Generation provenance is recorded in subset_comparison_summary.json."])
    (output_dir / "subset_comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frank_data", required=True)
    ap.add_argument("--device", default="cuda")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--compare_only", action="store_true", help="Recompute diagnostics from cached text without loading models")
    mode.add_argument("--regenerate", action="store_true", help="Explicitly overwrite generation caches")
    args = ap.parse_args()
    baseline_dir, output_dir = Path(args.baseline_dir), Path(args.output_dir)
    if (args.checkpoint or args.compare_only) and baseline_dir.resolve() == output_dir.resolve():
        ap.error("Baseline and finetuned output directories must differ")
    subset, annotations = build_bart_error_subset(args.frank_data)
    provenance = {}
    if args.compare_only:
        baseline, provenance["baseline"] = read_cached_for_comparison(baseline_dir)
        finetuned, provenance["finetuned"] = read_cached_for_comparison(output_dir)
    else:
        if args.checkpoint and not Path(args.checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
        baseline = load_or_generate(args.model_name, None, args.frank_data, baseline_dir, args.device, args.regenerate)
        if not args.checkpoint:
            print(f"Baseline generated/loaded: {len(baseline)} samples")
            return
        finetuned = load_or_generate(args.model_name, args.checkpoint, args.frank_data, output_dir, args.device, args.regenerate)
        _, provenance["baseline"] = read_cached_for_comparison(baseline_dir)
        _, provenance["finetuned"] = read_cached_for_comparison(output_dir)
    original_map = {row["hash"]: row for row in subset}
    for side, records in (("baseline", baseline), ("finetuned", finetuned)):
        indexed = index_results(records, side)
        if indexed.keys() != original_map.keys():
            raise ValueError(f"{side} does not cover the current BART-annotated article subset")
        for key, row in indexed.items():
            if any(row[field] != original_map[key].get(field, "") for field in ("article", "reference")):
                raise ValueError(f"{side} cached source/reference differs from FRANK data for {key}")
    provenance["current_frank_data_sha256"] = {name: file_sha256(Path(args.frank_data) / name)
                                                for name in ("benchmark_data.json", "human_annotations.json")}
    summary = compare(baseline, finetuned, output_dir, annotations, provenance)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
