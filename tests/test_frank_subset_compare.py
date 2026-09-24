import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import evaluate_frank_subset_compare as compare_module


def _generation_row(hash_value, article, reference, summary, **extra):
    row = {
        "hash": hash_value,
        "article": article,
        "reference": reference,
        "generated_summary": summary,
    }
    row.update(extra)
    return row


class TextScreeningTests(unittest.TestCase):
    def test_normalize_only_collapses_whitespace(self):
        self.assertEqual(compare_module.normalize("  Ann\n\tSaid, HELLO!  "), "Ann Said, HELLO!")
        self.assertEqual(compare_module.normalize(None), "")

    def test_lexically_absent_items_uses_literal_boundaries(self):
        article = "Joanne reported 50 cases in Paris."
        summary = "Ann reported 50 cases in Paris using 2024 data."

        absent = compare_module.lexically_absent_items(summary, article)

        self.assertIn("Ann", absent)
        self.assertIn("2024", absent)
        self.assertNotIn("50", absent)
        self.assertNotIn("Paris", absent)


class BartSubsetTests(unittest.TestCase):
    def _write_frank_data(self, directory, benchmark, annotations):
        directory.joinpath("benchmark_data.json").write_text(
            json.dumps(benchmark), encoding="utf-8"
        )
        directory.joinpath("human_annotations.json").write_text(
            json.dumps(annotations), encoding="utf-8"
        )

    def test_subset_uses_original_bart_record_and_keeps_metadata_separate(self):
        benchmark = [
            {
                "hash": "shared",
                "model_name": "pegasus",
                "article": "The article.",
                "reference": "The reference.",
                "summary": "Pegasus summary.",
            },
            {
                "hash": "shared",
                "model_name": "bart",
                "article": "The article.",
                "reference": "The reference.",
                "summary": "Original BART summary.",
            },
            {
                "hash": "unannotated",
                "model_name": "bart",
                "article": "Unused.",
                "reference": "Unused reference.",
                "summary": "Unused summary.",
            },
        ]
        annotation = {
            "hash": "shared",
            "model_name": "bart",
            "EntE": 0.0,
            "RelE": 1.0,
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            frank_dir = Path(raw_dir)
            self._write_frank_data(frank_dir, benchmark, [annotation])

            rows, ann_by_hash = compare_module.build_bart_error_subset(frank_dir)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_name"], "bart")
        self.assertEqual(rows[0]["summary"], "Original BART summary.")
        self.assertEqual(
            ann_by_hash["shared"],
            {
                "errors": ["EntE"],
                "has_bart_error": True,
                "annotation": annotation,
                "original_summary": "Original BART summary.",
            },
        )

    def test_subset_rejects_conflicting_bart_records_for_one_hash(self):
        benchmark = [
            {
                "hash": "duplicate",
                "model_name": "bart",
                "article": "Article one.",
                "reference": "Reference.",
                "summary": "Summary one.",
            },
            {
                "hash": "duplicate",
                "model_name": "bart",
                "article": "Article two.",
                "reference": "Reference.",
                "summary": "Summary two.",
            },
        ]
        annotations = [{"hash": "duplicate", "model_name": "bart", "EntE": 0.0}]
        with tempfile.TemporaryDirectory() as raw_dir:
            frank_dir = Path(raw_dir)
            self._write_frank_data(frank_dir, benchmark, annotations)

            with self.assertRaises(ValueError):
                compare_module.build_bart_error_subset(frank_dir)


class CompareValidationTests(unittest.TestCase):
    def setUp(self):
        self.article = "Joanne filed the report."
        self.reference = "A report was filed."
        self.base = _generation_row(
            "one", self.article, self.reference, "Joanne filed the report."
        )
        self.finetuned = dict(self.base)

    def _compare(self, baseline, finetuned):
        with tempfile.TemporaryDirectory() as raw_dir:
            return compare_module.compare(baseline, finetuned, Path(raw_dir))

    def test_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            self._compare([], [self.finetuned])
        with self.assertRaises(ValueError):
            self._compare([self.base], [])

    def test_rejects_duplicate_hashes_on_either_side(self):
        with self.assertRaises(ValueError):
            self._compare([self.base, dict(self.base)], [self.finetuned])
        with self.assertRaises(ValueError):
            self._compare([self.base], [self.finetuned, dict(self.finetuned)])

    def test_rejects_different_hash_sets(self):
        other = dict(self.finetuned, hash="other")
        with self.assertRaises(ValueError):
            self._compare([self.base], [other])

    def test_rejects_article_or_reference_mismatch(self):
        with self.assertRaises(ValueError):
            self._compare([self.base], [dict(self.finetuned, article="Changed article.")])
        with self.assertRaises(ValueError):
            self._compare([self.base], [dict(self.finetuned, reference="Changed reference.")])


class CacheIntegrityTests(unittest.TestCase):
    def test_default_cache_loading_rejects_missing_manifest_without_generating(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            compare_module.write_json(
                output_dir / compare_module.RESULTS_NAME,
                [_generation_row("one", "Article.", "Reference.", "Summary.")],
            )
            with mock.patch.object(
                compare_module, "generation_context", return_value={"schema_version": 2}
            ), mock.patch.object(compare_module, "generate_results") as generate:
                with self.assertRaises(ValueError):
                    compare_module.load_or_generate(
                        "model", None, output_dir, output_dir, "cpu"
                    )

            generate.assert_not_called()

    def test_compare_only_accepts_legacy_cache_as_unverified(self):
        rows = [_generation_row("one", "Article.", "Reference.", "Summary.")]
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            compare_module.write_json(output_dir / compare_module.RESULTS_NAME, rows)

            loaded, provenance = compare_module.read_cached_for_comparison(output_dir)

        self.assertEqual(loaded, rows)
        self.assertEqual(provenance["status"], "legacy_unverified")
        self.assertIn("results_sha256", provenance)

    def test_compare_only_rejects_results_modified_after_manifest(self):
        original = [_generation_row("one", "Article.", "Reference.", "Summary.")]
        modified = [_generation_row("one", "Article.", "Reference.", "Changed.")]
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            results_path = output_dir / compare_module.RESULTS_NAME
            compare_module.write_json(results_path, original)
            compare_module.write_json(
                output_dir / compare_module.MANIFEST_NAME,
                {
                    "context": {"schema_version": 2},
                    "results_sha256": compare_module.file_sha256(results_path),
                },
            )
            compare_module.write_json(results_path, modified)

            with self.assertRaises(ValueError):
                compare_module.read_cached_for_comparison(output_dir)

    def test_missing_annotation_scores_are_unknown_not_no_error(self):
        errors, status = compare_module.annotation_error_status(
            {"hash": "one", "model_name": "bart", "EntE": 1.0}
        )

        self.assertEqual(errors, [])
        self.assertIsNone(status)


class CompareOutputTests(unittest.TestCase):
    def test_schema_v2_transitions_and_frank_labels(self):
        article = "Joanne met Carol in Paris in 2020."
        reference = "A meeting took place."
        baseline = [
            _generation_row("to_clear", article, reference, "Ann met Carol."),
            _generation_row("to_flagged", article, reference, "Joanne met Carol."),
            _generation_row("stays_flagged", article, reference, "Ann met Carol."),
            _generation_row("stays_clear", article, reference, "Joanne met Carol."),
        ]
        finetuned = [
            _generation_row("to_clear", article, reference, "Joanne met Carol."),
            _generation_row("to_flagged", article, reference, "Bob met Carol."),
            _generation_row("stays_flagged", article, reference, "Bob met Carol."),
            _generation_row("stays_clear", article, reference, "Joanne met Carol."),
        ]
        annotations = {
            "to_clear": {
                "errors": ["EntE"],
                "has_bart_error": True,
                "annotation": {"annotator": "human"},
                "original_summary": "  Ann\nmet Carol. ",
            },
            "to_flagged": {
                "errors": [],
                "has_bart_error": False,
                "annotation": {"annotator": "human"},
                "original_summary": "Joanne met Carol.",
            },
            "stays_flagged": {
                "errors": ["EntE"],
                "has_bart_error": True,
                "annotation": None,
                "original_summary": "Ann met Carol.",
            },
            "stays_clear": {
                "errors": [],
                "has_bart_error": False,
                "annotation": {"annotator": "human"},
                "original_summary": "Joanne met Carol.",
            },
        }

        # Deliberately make fine-tuned summaries score much better. Lexical
        # transition flags must still depend only on literal entity/number checks.
        def fake_rouge(_reference, prediction):
            score = 0.95 if prediction.startswith("Bob") else 0.10
            return {"rouge1": score, "rouge2": score, "rougeL": score}

        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            with mock.patch.object(compare_module, "rouge_pair", side_effect=fake_rouge):
                returned = compare_module.compare(
                    baseline,
                    finetuned,
                    output_dir,
                    ann_by_hash=annotations,
                    provenance={"test_run": True},
                )

            summary = json.loads(
                output_dir.joinpath("subset_comparison_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            with output_dir.joinpath("subset_comparison.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            markdown = output_dir.joinpath("subset_comparison_summary.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(returned, summary)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["n_samples"], 4)
        self.assertEqual(
            summary["lexical_screen_transition_counts"],
            {
                "flagged_to_clear": 1,
                "clear_to_flagged": 1,
                "flagged_to_flagged": 1,
                "clear_to_clear": 1,
            },
        )
        self.assertNotIn("judgement_counts", summary)
        self.assertNotIn("likely_improved", json.dumps(summary))
        self.assertNotIn("likely_worse", json.dumps(summary))

        rows_by_hash = {row["hash"]: row for row in rows}
        self.assertEqual(rows_by_hash["to_clear"]["baseline_frank_label"], "error")
        self.assertEqual(rows_by_hash["to_clear"]["finetuned_frank_label"], "unknown")
        self.assertEqual(rows_by_hash["to_flagged"]["baseline_frank_label"], "no_error")
        self.assertEqual(rows_by_hash["to_flagged"]["finetuned_frank_label"], "unknown")
        self.assertEqual(rows_by_hash["stays_flagged"]["baseline_frank_label"], "unknown")
        self.assertEqual(rows_by_hash["stays_clear"]["baseline_frank_label"], "no_error")
        self.assertEqual(rows_by_hash["stays_clear"]["finetuned_frank_label"], "no_error")
        for row in rows:
            self.assertIn(row["baseline_frank_label"], {"error", "no_error", "unknown"})
            self.assertIn(row["finetuned_frank_label"], {"error", "no_error", "unknown"})

        self.assertIn("4", markdown)
        self.assertNotIn("Samples: 250", markdown)
        self.assertNotIn("46 error", markdown)
        self.assertNotIn("204 no-error", markdown)


if __name__ == "__main__":
    unittest.main()
