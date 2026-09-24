# Temporal Brain–Model Alignment for Factuality-Related Linguistic Cues in Generative Language Models

Research code accompanying **“Temporal Brain-Model Alignment for Factuality-Related Linguistic Cues in Generative Language Models”** by Wenqing Zhou, Zhejun Zhang, Shaoting Guo, Lin Zhang, and Lei Li.

This release currently contains the EEG time-window tuning pipeline for BART-large and PEGASUS-large. It uses word-level DERCo EEG as supervision for source-side encoder representations and keeps the decoder frozen. The repository does **not** redistribute EEG data, FRANK data, pretrained model weights, checkpoints, or generated summaries.

## Method implemented here

For each subject and fold, the encoder receives the causal source context through the current word. The final-layer hidden states of the current word's subword tokens are mean-pooled, then mapped to an EEG sensor vector by a lightweight projection head. Training minimizes mean-squared error between that prediction and a time-window-averaged EEG topography.

The paper-defined windows are:

| Word category | EEG window |
|---|---:|
| Named entity | 150–350 ms |
| Noun or verb | 0–500 ms |
| Other | Mean over the available epoch |

The decoder is frozen and no summarization loss or FRANK factual label is used during EEG tuning. FRANK is used only for paired baseline-versus-tuned evaluation.

The checked-in learning rates, epoch limit, projection-head architecture, dropout, batching, optimizer, and scheduler are engineering settings recovered from the training project; they should not be interpreted as paper-defined methodological constants. The public batch defaults are deliberately conservative to reduce out-of-memory risk.

## Repository layout

```text
configs/train/                         BART-large and PEGASUS-large configs
scripts/train.py                       DERCo/FRANK EEG training entry point
scripts/evaluate_frank.py              baseline or tuned summary generation
scripts/evaluate_frank_subset_compare.py
scripts/compute_frank_metrics.py       descriptive evaluation utilities
scripts/analyze_frank_by_errors.py
src/data/                              stimulus, event, EEG, and token alignment
src/models/                            encoder wrappers and EEG mapper
src/losses/                            MSE and optional exploratory losses
src/training/                          training, validation, and checkpointing
tests/                                 safety and time-window tests
```

## Installation

Python 3.9 or newer is supported; Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m spacy download en_core_web_sm
```

The default configurations download `facebook/bart-large-cnn` or `google/pegasus-large` from Hugging Face. To run fully offline, set `model.model_name` to a trusted local model directory.

## Data preparation

Obtain DERCo from its official distribution and follow its terms and ethics requirements. This code expects the preprocessed layout below; preprocessing arrays are intentionally absent from Git:

```text
data/EEG-Data-Derco/
├── article_0.pkl
├── ...
├── article_4.pkl
└── SUBJECT_ID/
    ├── article_0/
    │   ├── preprocessed_epoch.npy   # epochs × 32 channels × time
    │   └── event_info.json          # includes event_numbers
    └── ...
```

`event_numbers` are used explicitly to map EEG epochs to words; array row numbers are not treated as word indices. The loader validates dimensions, event counts, event bounds, and channel counts before training. DERCo stimulus pickles are loaded by a restricted data-only unpickler because ordinary pickle loading can execute code.

The recovered project used 1000 Hz DERCo epochs with a 200 ms pre-stimulus baseline. If your preprocessing differs, update `sampling_rate` and `baseline_ms` rather than reusing the defaults blindly.

## Training

BART-large:

```bash
python scripts/train.py \
  --config configs/train/bart_large_derco_timewindow.yaml \
  --subject SUBJECT_ID \
  --n_folds 5 \
  --fold 0
```

PEGASUS-large:

```bash
python scripts/train.py \
  --config configs/train/pegasus_large_derco_timewindow.yaml \
  --subject SUBJECT_ID \
  --n_folds 5 \
  --fold 0
```

Configuration values can be overridden without editing YAML:

```bash
python scripts/train.py \
  --config configs/train/bart_large_derco_timewindow.yaml \
  --subject SUBJECT_ID \
  --set data.eeg_data_dir=/absolute/path/to/EEG-Data-Derco \
  --set data.stimulus_dir=/absolute/path/to/EEG-Data-Derco \
  --set training.batch_size=1
```

Checkpoints contain trainable tensors only and are written under `outputs/`, which is ignored by Git.

## FRANK evaluation

Generate a baseline or load a trusted EEG-tuned checkpoint:

```bash
python scripts/evaluate_frank.py \
  --model_name facebook/bart-large-cnn \
  --frank_data /absolute/path/to/frank/data \
  --output_dir outputs/frank_eval/baseline

python scripts/evaluate_frank.py \
  --model_name facebook/bart-large-cnn \
  --checkpoint outputs/bart_large_derco_timewindow/SUBJECT_ID/fold_0/best_model.pt \
  --frank_data /absolute/path/to/frank/data \
  --output_dir outputs/frank_eval/tuned
```

PyTorch checkpoints are loaded with `weights_only=True`. Do not weaken this safeguard for an untrusted checkpoint.

The scripts provide generation and descriptive comparison support. They do not reconstruct unavailable human judgments, unpublished data splits, or missing experimental metadata. Reproducing paper-level numerical results requires the same authorized datasets, preprocessing, subject/fold definitions, decoding settings, and evaluation annotations.

## Verification

```bash
python -m compileall -q src scripts tests
pytest -q
```

Before publishing a change, also confirm that no dataset, weight, checkpoint, output, credential, server path, or participant-level result is staged:

```bash
git status --short
git diff --cached --stat
```

## Safety and privacy

See [SECURITY.md](SECURITY.md). The release excludes internal batch scripts, destructive output-rotation commands, machine-specific paths, logs, datasets, model weights, checkpoints, and result tables. Never commit participant data or credentials. Review the license and consent terms of every external dataset before use or redistribution.

## Provenance and limitations

The tuning implementation was restored from a research training snapshot and sanitized for public release. The earlier repository named `RSA-TCA` was unavailable during this migration, so no code or README text from that repository was copied. This release therefore covers the verified EEG tuning workflow, not a claimed byte-for-byte migration of unavailable code.

No software license has been selected for this repository. Until the authors add one, copyright law reserves reuse rights by default.

## Citation

The formal citation will be added when the paper metadata is publicly available. Until then, please cite the paper by title and authors listed above.
