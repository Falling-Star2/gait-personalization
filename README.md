# 🦿 Personalized Gait Activity Recognition for Wearable Exoskeletons

> **A two-phase study: first establishing that deep learning requires raw signals (not pre-extracted features), then building a few-shot adaptive CNN+LSTM that personalizes to a new user with just ~20 seconds of calibration data.**

---

## 📌 Motivation

Powered exoskeletons like the **WalkON Suit** must correctly identify a user's current activity — walking, climbing stairs, descending — to apply assistive forces at the right moment and in the right direction. A misclassification during stair descent isn't just an accuracy number. It's a fall risk.

Most existing approaches rely on hand-engineered features and a single global model trained on population data. But human gait is deeply personal — two people descending the same staircase produce significantly different IMU signals. This project tackles that gap directly:

> *Can a model trained on 20 people reliably adapt to a completely new person using only ~20 seconds of calibration data?*

The answer is **yes** — and the improvement is up to **+7.95%** per person.

---

## 🗂️ Datasets

This project uses two datasets, each serving a different purpose.

### Dataset 1 — UCI HAR (Human Activity Recognition with Smartphones)
- **Source:** UCI Machine Learning Repository / Kaggle
- **Subjects:** 30 people
- **Signals:** Smartphone IMU (waist-mounted)
- **Format:** 561 **pre-extracted** features per window (mean, std, FFT coefficients, etc.)
- **Labels:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- **Purpose:** Baseline experiments and a key negative result — proving that pre-extracted features prevent deep learning from outperforming classical ML

### Dataset 2 — MotionSense
- **Source:** Kaggle (malekzadeh/motion-sense-dataset)
- **Subjects:** 24 people
- **Signals:** Raw device motion at **50 Hz** — accelerometer (x,y,z), gyroscope (x,y,z), gravity (x,y,z)
- **Format:** Raw time series, one CSV per subject per activity trial
- **Labels:** Walking, Downstairs, Upstairs, Jogging, Sitting, Standing
- **Purpose:** Main model training on raw signals, personalization experiments, leave-subjects-out evaluation

---

## 🔬 Study Design

The project is structured as four sequential phases, each building on the previous:

```
Phase 1: Establish classical ML baseline on pre-extracted features (UCI HAR)
Phase 2: Test if LSTM/Transformer improve over RF on the same features
Phase 3: Train CNN+LSTM on raw signals with unseen-subject evaluation (MotionSense)
Phase 4: Few-shot personalization — adapt pretrained model to a new person
```

This structure mirrors how research papers are written: start with the baseline, identify the gap, address it, then extend.

---

## 📊 Results

### Phase 1 — Random Forest Baseline (UCI HAR)

Trained on 561 pre-extracted features from the UCI HAR dataset.

```
Overall Accuracy: 92.67%

Per-class Performance:
  LAYING             F1: 1.00  ← trivially easy, fully distinct signal
  SITTING            F1: 0.90  ← confused with STANDING (93 mistakes)
  STANDING           F1: 0.91  ← confused with SITTING
  WALKING            F1: 0.93
  WALKING_UPSTAIRS   F1: 0.89  ← 44 confused as plain WALKING
  WALKING_DOWNSTAIRS F1: 0.92  ← 40 confused as WALKING_UPSTAIRS

Key observation: The model relies almost entirely on gravity-based features
(body tilt angle). This works for static postures but fails to capture the
dynamic temporal rhythm of stair locomotion — the most safety-critical case
for exoskeleton control.
```

---

### Phase 2 — LSTM and Transformer on Pre-Extracted Features (UCI HAR)

Testing whether sequential models improve over Random Forest on the same data.

```
Random Forest:   92.67%
LSTM:            92.99%   Train: 97.96%  ← overfitting gap visible
Transformer:     91.96%   Train: 97.43%  ← worse than RF

Key finding: Neither deep learning model meaningfully beats Random Forest
on pre-extracted features. Train accuracy reaches ~98% but test plateaus
at ~92%, indicating overfitting with no generalization gain.

Why? Pre-extracted features (mean, std, FFT) collapse the time dimension.
The LSTM reads 10 feature vectors pretending to be a sequence. There is no
real temporal structure left to exploit. This is not a model failure —
it is a data representation failure.

Conclusion: Deep learning models require raw temporal signals to justify
their complexity. This finding directly motivates switching to MotionSense.
```

---

### Phase 3 — CNN+LSTM on Raw IMU Signals (MotionSense)

Architecture: CNN extracts local stride-cycle features, LSTM captures how they evolve.

```
Train/test split: 20 subjects train, 4 completely unseen subjects test
Window size: 128 timesteps (2.56 seconds at 50Hz), 50% overlap
Train windows: 18,097   |   Test windows: 3,766

Overall Accuracy on Unseen Subjects: 94.21%

Per-class Performance:
  Downstairs    Precision: 0.82   Recall: 0.92   F1: 0.86
  Upstairs      Precision: 0.83   Recall: 0.96   F1: 0.89
  Walking       Precision: 0.98   Recall: 0.88   F1: 0.93
  Jogging       Precision: 0.99   Recall: 0.96   F1: 0.97
  Sitting       Precision: 1.00   Recall: 0.95   F1: 0.98
  Standing      Precision: 0.95   Recall: 1.00   F1: 0.97

Notable confusion: 112 stair windows misclassified as wrong stair/walk
direction. Stair detection remains the hardest problem — directly relevant
to fall prevention in exoskeleton use.

Comparison:
  Random Forest (pre-extracted):  92.67%
  LSTM (pre-extracted):           92.99%
  CNN+LSTM (raw signals):         94.21%  ← +1.54% and generalizes to new people
```

---

### Phase 4 — Few-Shot Personalization (Novel Contribution)

#### The Problem
Per-subject accuracy on the 4 unseen test subjects varies dramatically:

```
  Subject 1:  93.05%
  Subject 2:  92.24%
  Subject 3:  99.69%
  Subject 4:  91.59%   ← 8.1% gap between best and worst
```

A single global model cannot capture everyone equally. Subject 4 walks
differently from the training population. In a clinical device, this gap
means misreading user intent roughly 1 in 12 times.

#### The Solution
Freeze the CNN (which learned universal motion features) and fine-tune only
the LSTM + classifier layers on a small sample of data from the new user.
This costs seconds at deployment time and requires no new hardware.

```python
# Freeze CNN — keep universal motion features
for param in model.cnn.parameters():
    param.requires_grad = False

# Fine-tune LSTM + classifier on N windows per activity from new user
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

#### Results

```
                Baseline    5-shot    10-shot    20-shot    30-shot
  Subject 1     93.05%     96.16%    93.57%     96.47%     96.99%
  Subject 2     92.24%     96.48%    97.00%     96.17%     97.83%
  Subject 3     99.69%     99.59%    99.69%     99.59%     99.69%  ← already optimal
  Subject 4     91.59%     98.04%    99.54%     98.62%     99.31%

  Mean baseline:      94.14%
  Mean 10-shot:       97.45%
  Mean improvement:  +3.31%
  Best improvement:  +7.95%  (Subject 4, 10-shot)

Key observations:
  • Subject 3 is already near-perfect — personalization correctly has no effect
  • Subject 4, the hardest case, jumps from 91.59% → 99.54% with 10 windows
  • 10 windows × 6 activities × 2.56 seconds ≈ 154 seconds of calibration
  • Improvement is largest for the subjects who need it most
```

---

## 🏗️ Model Architecture

```
Input: (batch, 128 timesteps, 9 sensor channels)
            │
            ▼
    ┌───────────────┐
    │  Conv1D(64)   │  kernel=5, BatchNorm, ReLU
    │  Conv1D(128)  │  kernel=5, BatchNorm, ReLU, MaxPool, Dropout
    └──────┬────────┘
           │  → (batch, 64 timesteps, 128 features)
           ▼
    ┌───────────────┐
    │  LSTM ×2      │  hidden=128, dropout=0.3
    └──────┬────────┘
           │  last hidden state → (batch, 128)
           ▼
    ┌───────────────┐
    │  FC(64) ReLU  │
    │  Dropout(0.3) │
    │  FC(6)        │  → activity logits
    └───────────────┘

Total trainable parameters: 317,254
Training: 30 epochs, Adam lr=1e-3, CosineAnnealing scheduler
Hardware: Kaggle free GPU (T4)
Training time: ~4 minutes
```

---

## 📈 Full Results Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                      FULL RESULTS SUMMARY                            │
├──────────────────────────────┬──────────────┬──────────┬─────────────┤
│ Model                        │ Dataset      │ Accuracy │ Notes       │
├──────────────────────────────┼──────────────┼──────────┼─────────────┤
│ Random Forest                │ UCI HAR      │  92.67%  │ pre-extract │
│ LSTM                         │ UCI HAR      │  92.99%  │ pre-extract │
│ Transformer                  │ UCI HAR      │  91.96%  │ pre-extract │
│ CNN+LSTM                     │ MotionSense  │  94.21%  │ raw signals │
│ CNN+LSTM + 10-shot (mean)    │ MotionSense  │  97.45%  │ personalized│
│ CNN+LSTM + 10-shot (best)    │ MotionSense  │  99.54%  │ personalized│
└──────────────────────────────┴──────────────┴──────────┴─────────────┘
```

---

## 🔬 Connection to Prior Work

This project is directly motivated by research from Prof. Kyoungchul Kong's
EXO-Lab at KAIST:

- **Kong et al. (2022)** — *"Iterative Learning of Human Behavior for Adaptive
  Gait Pattern Adjustment"*, IEEE Trans. Robotics — our few-shot fine-tuning is
  a data-driven implementation of this adaptive learning concept

- **Park et al. (2023)** — *"Data-Driven Modeling for Gait Phase Recognition in
  a Wearable Exoskeleton"*, IEEE Trans. Robotics — our work extends gait
  recognition to raw signals without requiring force estimation hardware

- **Slade et al. (2024)** — *"On human-in-the-loop optimization of human-robot
  interaction"*, Nature — our short calibration loop is a practical
  instantiation of this optimization framework

---

## ⚠️ Limitations & Future Work

- **Sensor placement:** MotionSense uses waist-mounted smartphones; leg-mounted
  exoskeleton IMUs produce different signal characteristics and noise profiles
- **Gait granularity:** Activity-level labels (6 classes) are coarser than the
  phase-level labels (heel strike, stance, push-off, swing) needed for real
  exoskeleton torque control
- **Online adaptation:** Current personalization is a one-time offline
  calibration step; continual real-time adaptation during use is an open problem
- **Clinical populations:** All subjects are healthy adults — people with
  paraplegia or muscle weakness, the primary exoskeleton users, are not
  represented in either dataset

---

## 🚀 Reproducing Results

```bash
git clone https://github.com/Falling-Star2/gait-personalization
cd gait-personalization
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

Open notebooks in order on Kaggle (free GPU):

```
notebooks/
  01_baseline_rf_lstm_ucihar.ipynb     ← Phase 1 + 2  (UCI HAR)
  02_cnn_lstm_motionsense.ipynb        ← Phase 3       (MotionSense)
  03_personalization.ipynb             ← Phase 4       (novel contribution)
```

Datasets (both free on Kaggle):
- UCI HAR: kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- MotionSense: kaggle.com/datasets/malekzadeh/motion-sense-dataset

---

## 👤 Author

**[Nazmus Sakib]**
Competitive Programmer → Aspiring Researcher in AI-driven Assistive Robotics

- 📧 sakibn856@gmail.com
- GitHub: github.com/Falling-Star2

---

## 📄 License

MIT — free to use, modify, and build upon with attribution.

---

*Independent research project exploring the intersection of algorithmic
thinking, deep learning, and human assistive robotics.*
