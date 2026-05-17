# Garbage Image Classification with MobileNetV2

A multiclass image classifier for 12 categories of household waste, trained with transfer learning on the [Garbage Classification dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) (Kaggle, 15,515 images).

## Results

Evaluated on a held-out test split of 2,325 images:

| Metric | Score |
|---|---|
| Accuracy | 94.88% |
| Macro F1 | 0.929 |
| Balanced accuracy | 92.98% |
| Top-2 accuracy | 98.45% |
| ROC/AUC (OvR macro) | 0.998 |

For reference, the majority-class baseline (DummyClassifier) on the same test split scored 34.4% accuracy and 0.043 macro F1.

![Per-class F1 score](outputs/garbage_classifier_per_class_f1.png)

The model is weakest on white-glass (F1 0.85), plastic, and metal (both 0.87). These classes are visually ambiguous even at human inspection.

## Why this project

I wanted a side project that wasn't another Titanic clone. Garbage classification is practically relevant (sorting at recycling facilities), the class distribution is imbalanced enough to punish lazy "accuracy-only" thinking, and several classes share enough visual features that the model occasionally guesses wrong with high confidence. That last failure mode is the one worth studying.

The notebook is also structured to serve as my own reference for image classification methodology: leakage checks, baseline comparison, stratified split, two-stage training, multi-metric evaluation, error analysis, and interpretability.

## Methodology

The pipeline, in order:

1. **Dataset acquisition** with `kagglehub`, Kaggle CLI fallback
2. **Folder structure detection** that doesn't assume class paths
3. **Metadata and validation** — every file has a label, classes meet minimum support
4. **Exact-duplicate check** via MD5 hash across all images
5. **Stratified 70/15/15 split** at the file level, with post-split hash-based leakage verification
6. **`tf.data` pipeline** with augmentation kept as model layers (auto-disabled at inference)
7. **Class weights** computed from train data only
8. **Baseline** — DummyClassifier, `most_frequent` strategy
9. **Stage 1 training** — frozen MobileNetV2 backbone, classifier head only (8 epochs, lr=1e-3)
10. **Stage 2 training** — unfreeze last 35 layers, BatchNorm stays frozen, lr=1e-5, 6 epochs
11. **Evaluation** — accuracy, balanced accuracy, macro/weighted F1, top-2, ROC/AUC OvR
12. **Error analysis** — classification report, confusion matrix, confidence distributions
13. **Grad-CAM** applied to high-confidence wrong predictions
14. **Export** — model `.keras`, label mapping, config, metrics summary, predictions, plots

A few choices worth flagging:

- **MD5 for leakage detection.** Image datasets often have duplicate files. Without this check, accuracy gets inflated quietly. MD5 only catches exact duplicates though, not near-duplicates (see Limitations).
- **Baseline before model.** A model that beats DummyClassifier by 60 percentage points is doing real work. Without the baseline, "94% accuracy" is a number without context.
- **Class weights from train only.** Computing weights using val/test data is a subtle form of leakage. Easy to miss.
- **Macro F1 over accuracy.** Roughly 25% of the dataset is "clothes." A model that nails clothes and fails on minorities can still hit 94% accuracy. Macro F1 treats every class equally.
- **BatchNorm frozen during fine-tuning.** Unfreezing BN during transfer learning can destabilize pretrained statistics.

## Project structure

```
garbage-image-classification/
├── notebooks/
│   └── garbage_classification_mobilenetv2.ipynb
├── outputs/
│   └── garbage_classifier_per_class_f1.png
├── requirements.txt
├── .gitignore
└── README.md
```

## How to reproduce

The notebook is designed for Run All execution on Google Colab or Kaggle.

```bash
# Local setup
pip install -r requirements.txt
jupyter notebook notebooks/garbage_classification_mobilenetv2.ipynb
```

Dataset is downloaded automatically via `kagglehub`. If that fails, place your `kaggle.json` in `~/.kaggle/` for the CLI fallback.

Approximate training time on a Colab T4 GPU: 25 minutes for the full pipeline.

## Limitations

Things I didn't do that I'd want to before calling this "production":

- **Test on real-world dirty photos.** The Kaggle dataset is studio-clean. Performance on actual sorting-facility images would almost certainly drop.
- **Near-duplicate detection.** MD5 only catches identical bytes. Perceptual hashing (pHash) or embedding-based similarity would catch resized or recompressed copies.
- **Confidence calibration.** The model is sometimes confidently wrong. Temperature scaling or isotonic regression would help before any threshold-based decision.
- **Cross-dataset validation.** Train on one dataset, test on another. That's the actual test of generalization.
- **Architecture comparison.** Only tried MobileNetV2. EfficientNet, ConvNeXt, or a ViT would give a sense of where the ceiling actually is.

## Dataset

[Garbage Classification by Mostafa Abla](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) on Kaggle. 15,515 images across 12 classes: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass.

Pretrained MobileNetV2 weights from Keras Applications (originally trained on ImageNet).

---

Built by [Muhammad Faiz Arkan Herwanto](https://github.com/PaezArkan).
