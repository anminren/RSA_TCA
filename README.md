# Temporal Brain–Model Alignment for Factuality-Related Linguistic Cues in Generative Language Models

Code accompanying **“Temporal Brain-Model Alignment for Factuality-Related Linguistic Cues in Generative Language Models”** by Wenqing Zhou, Shaoting Guo, Lin Zhang, and Lei Li.

This repository contains both parts of the project:

1. the original **RSA-TCA analysis code** from the `RSA-TCA-1.0.0` release, including hidden-state extraction, time-course representational similarity analysis, factuality-conditioned analysis, significance testing, layer selection, and significant-area calculation;
2. the **EEG time-window tuning pipeline** for BART-large and PEGASUS-large.

The repository does not redistribute EEG recordings, participant-level arrays, pretrained weights, checkpoints, FRANK annotations, or generated experiment outputs.

## Repository layout

```text
experiment_with_notes/                 Original RSA-TCA analysis code
├── get_hidden_data_allModel.py        Extract encoder/decoder hidden states
├── n_timepoints_anlysis.py            Time-course RSA at configurable resolution
├── faculty_analysis.py                Factuality-conditioned RSA-TCA
├── significance.py                    Wilcoxon tests over RSA time courses
├── factuality_significance.py         Factuality-conditioned significance tests
├── choose_layer.py                    Select layers from significant RSA scores
├── factuality_choose_layer.py         Factuality-conditioned layer selection
├── SigArea.py                         Area over significant time points
└── xsum/                              Selected source/target stimuli from the release

configs/train/                         BART-large and PEGASUS-large tuning configs
scripts/train.py                       DERCo EEG tuning entry point
scripts/evaluate_frank.py              Baseline or tuned summary generation
scripts/evaluate_frank_subset_compare.py
scripts/compute_frank_metrics.py       Descriptive evaluation utilities
scripts/analyze_frank_by_errors.py
src/data/                              Stimulus, event, EEG, and token alignment
src/models/                            Encoder wrappers and EEG mapper
src/losses/                            Tuning objectives
src/training/                          Training, validation, and checkpointing
tests/                                 Safety, alignment, and smoke tests
```

## Installation

Python 3.9 or newer is supported; Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
python -m spacy download en_core_web_sm
```

The scripts can download public models from Hugging Face. For an offline run, pass a trusted local model directory instead. Review model and dataset licenses before use.

## Part I: RSA-TCA analysis

### Expected data layout

The analysis scripts retain the relative layout used by the original release:

```text
hidden_states/
├── bart_base_encoder_align.npy
├── bart_encoder_align.npy
└── ...
npy_bart2/
├── art_1/
│   ├── Overall/sub_1_art_1_epo.npy
│   ├── N400/...
│   └── P600/...
└── ...
spearman_corr/
├── spearman/
└── p_value/
significance/
factuality_significance/
plots/
```

The large hidden-state, EEG, intermediate, and result arrays shown above are intentionally not committed. Obtain the relevant data through its authorized distribution and preserve its consent, ethics, and license restrictions.

### Procedure

1. Prepare a compatible encoder-decoder model. Models used in the original experiment included BART, PEGASUS, and T5 variants. Two model identifiers recorded in the release were `sysresearch101/t5-large-finetuned-xsum-cnn` and `jordiclive/flan-t5-3b-summarizer`; availability and exact revisions may change.

2. From `experiment_with_notes/`, extract hidden states:

   ```python
   from get_hidden_data_allModel import get_hidden_data

   get_hidden_data("trusted/model/path-or-id")
   ```

   The original study manually aligned each model's tokens to the PEGASUS tokenization and stored the result as `{model_name}_encoder_align.npy`.

3. Compute the overall or factuality-conditioned RSA time course:

   ```python
   from n_timepoints_anlysis import n_timepoints_analysis
   from faculty_analysis import faculty_analysis

   n_timepoints_analysis(model_name, n_timepoints)
   faculty_analysis(model_name, n_timepoints, factuality_type)
   ```

4. Run `all_run()` from `significance.py` or `factuality_significance.py` after setting the cases at the top of the script.

5. Run `choose_layer()` from `choose_layer.py` or `factuality_choose_layer.py`, then use `SigArea.py` for significant-area summaries.

These scripts are research workflows rather than a one-command benchmark. Check array shapes, model revisions, manual token alignments, subject definitions, and analysis choices before interpreting or comparing results.

## Part II: EEG time-window tuning

For each subject and fold, the encoder receives the causal source context through the current word. Final-layer hidden states for the current word's subword tokens are mean-pooled and mapped to an EEG sensor vector. Training minimizes mean-squared error against a time-window-averaged EEG topography.

| Word category | EEG window |
|---|---:|
| Named entity | 150–350 ms |
| Noun or verb | 0–500 ms |
| Other | Mean over the available epoch |

The decoder is frozen. No summarization loss or FRANK factual label is used during EEG tuning; FRANK is used only for paired baseline-versus-tuned evaluation. Learning rates, epoch limits, projection architecture, dropout, optimizer, and scheduler in the checked-in configs are recovered engineering settings, not paper-defined constants.

### DERCo preparation

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

`event_numbers` map EEG epochs to words; array row numbers are not treated as word indices. The loader validates dimensions, event counts, event bounds, channel counts, and target alignment. DERCo stimulus pickles use a restricted data-only unpickler because unrestricted pickle loading can execute code.

The recovered setup used 1000 Hz epochs with a 200 ms pre-stimulus baseline. If your preprocessing differs, change `sampling_rate` and `baseline_ms`.

### Training

```bash
python scripts/train.py \
  --config configs/train/bart_large_derco_timewindow.yaml \
  --subject SUBJECT_ID \
  --n_folds 5 \
  --fold 0

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

### FRANK evaluation

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

PyTorch checkpoints are loaded with `weights_only=True`; do not weaken this safeguard for an untrusted checkpoint. Reproducing paper-level results requires the same authorized data, preprocessing, subject/fold definitions, decoding settings, and annotations.

## Verification and safety

```bash
python -m compileall -q experiment_with_notes src scripts tests
pytest -q
```

See [SECURITY.md](SECURITY.md). Internal batch scripts, destructive output-rotation commands, machine-specific paths, credentials, logs, datasets, weights, checkpoints, and participant-level results are excluded. Never commit participant data or secrets.

No software license was present in the supplied `RSA-TCA-1.0.0` archive, and none has been inferred here. Until the authors add one, copyright law reserves reuse rights by default.

## Citation

Formal publication metadata will be added when available. Until then, cite the paper by the title and authors shown above.
