# PulseVision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![SciPy](https://img.shields.io/badge/SciPy-1.13-blue?style=flat-square&logo=scipy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Educational-purple?style=flat-square)

An end-to-end remote photoplethysmography (rPPG) pipeline built from scratch in Jupyter notebooks. PulseVision extracts a pulse waveform and estimates heart rate directly from a face video, no contact, no specialized hardware, just a standard RGB camera recording.

This project was built as an educational exploration of both classical signal processing and deep learning approaches to the rPPG problem, using a single subject video from the UBFC-rPPG dataset.

---

## What is rPPG

Remote photoplethysmography is a contactless method of measuring physiological signals from ordinary video by analysing subtle color changes in skin caused by blood volume variation during each heartbeat. When the heart pumps, the amount of blood under the skin changes slightly, causing tiny variations in the light reflected from the face. A camera captures these variations across consecutive frames, and image processing or deep learning methods extract a time-domain signal from them. From this signal, heart rate, pulse waveform, and related physiological indicators can be estimated.

The key physical insight is that hemoglobin absorbs green light (500-570nm) more strongly than red or blue. This makes the green channel the most sensitive to blood volume changes. Red light penetrates too deep and passes through vessels with low sensitivity; blue light barely penetrates the epidermis. Green light reaches the dermis where capillary networks are dense, making green channel fluctuations the primary carrier of the rPPG signal.

---

## Dataset

**UBFC-rPPG** (University of Burgundy Franche-Comte)

- 42 subjects, frontal face video recorded at 30fps, 640x480 resolution
- Ground truth from CMS50E finger pulse oximeter (PPG waveform + instantaneous HR)
- Subjects performed a mental arithmetic task to naturally elevate heart rate
- Publicly available at: https://sites.google.com/view/ybenezeth/ubfcrppg

This project uses a single subject (`subject1`) for the full pipeline demonstration:

| Property | Value |
|---|---|
| FPS | 29.26 |
| Total frames | 1547 |
| Duration | 52.86 seconds |
| Mean HR (ground truth) | 106.7 BPM |
| HR range | 97 – 113 BPM |

---

## Project Structure

```
PulseVision/
│
├── data/
│   └── subject1/
│       ├── vid.avi              — face video recording
│       └── ground_truth.txt     — PPG waveform and HR ground truth
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_face_roi_extraction.ipynb
│   ├── 03_rgb_signal_extraction.ipynb
│   ├── 04_signal_processing.ipynb
│   ├── 05_heart_rate_estimation.ipynb
│   ├── 06_Data_Preparation.ipynb
│   └── 07_Model.ipynb
│
├── outputs/
│
├── utils/
│   ├── __init__.py
│   ├── video_utils.py
│   ├── face_utils.py
│   └── signal_utils.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Overview

The full pipeline runs across seven notebooks in sequence. Each notebook is self-contained and saves its outputs to the `outputs/` directory for the next stage to consume.

```
vid.avi
   |
   |  [NB01] Data Exploration
   |  Load video metadata, visualize sample frames,
   |  plot ground truth PPG waveform and HR signal
   |
   |  [NB02] Face ROI Extraction
   |  Haar cascade face detection per frame
   |  Forehead bounding box = top 30% of face, center 60% width
   |  Save (x1, y1, x2, y2) per frame → roi_coords.npy
   |
   |  [NB03] RGB Signal Extraction
   |  Crop forehead patch per frame using saved ROI coordinates
   |  Spatial average of R, G, B pixel values per patch
   |  Build three 1D time series: R(t), G(t), B(t)
   |
   |  [NB04] Signal Processing — CHROM Algorithm
   |  Detrend each channel (remove slow drift)
   |  Bandpass filter 0.7–4.0 Hz (42–240 BPM range)
   |  CHROM: combine channels to cancel specular reflection noise
   |  Output: normalized rPPG waveform
   |
   |  [NB05] Heart Rate Estimation
   |  FFT on rPPG signal → frequency spectrum
   |  Peak frequency in 0.7–4.0 Hz → BPM estimate
   |  Sliding window HR over time (10s window, 1s step)
   |
   |  [NB06] Deep Learning Data Preparation
   |  Resize each ROI patch to 32x32
   |  Sliding window clips of T=30 frames, stride=5
   |  Pair clips with GT PPG windows as labels
   |  Time-based 80/20 train/val split
   |
   |  [NB07] 3D CNN Model — PhysNetSmall
   |  Train spatiotemporal convolutional network
   |  Pearson correlation loss
   |  Compare DL prediction vs CHROM vs ground truth
   |
Heart Rate (BPM) + rPPG Waveform
```

---

## Notebooks

### NB01 — Data Exploration

Loads the video and ground truth file to understand the data. Displays sample frames at regular intervals, decomposes a single frame into R/G/B channels, and plots the full-length PPG waveform with a zoom inset. Establishes the ground truth mean HR (106.7 BPM) as the target for subsequent methods.

Key observation: whole-frame pixel averages show G and B dominant over R due to background interference, motivating the need for ROI extraction in the next stage.

### NB02 — Face ROI Extraction

Uses OpenCV's Haar cascade frontal face detector to locate the face bounding box in every frame. The forehead ROI is defined geometrically from this box:

```
x1 = face_x + 20% of face_width
x2 = face_x + 80% of face_width
y1 = face_y + 2%  of face_height
y2 = face_y + 30% of face_height
```

The 20-80% horizontal crop avoids hairline edges and temporal regions with greater motion. The 2-30% vertical crop sits below the hairline and above the eyebrows. Saves `roi_coords.npy` containing 1547 bounding box tuples.

Effect on pixel statistics: R channel mean rises from 89 (whole frame) to 187 (forehead only), confirming the background was suppressing skin channel values.

### NB03 — RGB Signal Extraction

For each frame, crops the forehead patch using saved coordinates and computes the mean pixel value of each channel. This spatial averaging operation reduces the 2D patch to three scalar values per frame, producing R(t), G(t), B(t) time series of length 1547.

Key observation: G and B channels are highly correlated — they move almost identically across time. R behaves differently. This channel structure is predicted by rPPG theory and is the basis for the CHROM algorithm's noise cancellation in the next stage.

A motion artifact is visible as a sharp spike across all channels at approximately t=32 seconds, corresponding to a brief head movement.

### NB04 — Signal Processing

Applies a three-stage signal chain to extract the rPPG waveform.

**Detrending** removes low-frequency drift by subtracting a linear trend from each channel, eliminating slow illumination changes that would otherwise distort the signal.

**Bandpass filtering** applies a 4th-order Butterworth bandpass filter between 0.7 Hz and 4.0 Hz (corresponding to 42–240 BPM). This removes all content outside the physiologically plausible heart rate range, including the motion artifact's DC component and high-frequency camera noise.

**CHROM algorithm** combines the filtered channels using the following formulation:

```
Rn = R / mean(R),   Gn = G / mean(G),   Bn = B / mean(B)

X = 3*Rn - 2*Gn                         (pulse + specular direction)
Y = 1.5*Rn + Gn - 1.5*Bn               (specular noise direction)

alpha = std(X) / std(Y)                  (noise scale factor)

rPPG = X - alpha * Y                     (pulse with noise cancelled)
```

CHROM works by finding two orthogonal directions in the normalized RGB color space. The first (X) captures both pulse and specular reflection noise. The second (Y) captures primarily the specular direction. Subtracting the scaled specular component from X isolates the pulse signal.

### NB05 — Heart Rate Estimation

Applies the Fast Fourier Transform to the rPPG signal to identify the dominant oscillation frequency. The FFT converts the time-domain signal into a frequency spectrum; the highest-magnitude frequency component within the 0.7–4.0 Hz range is the estimated heart rate.

**Global estimate**: FFT over the full 52-second signal identifies a peak at 1.778 Hz = 106.7 BPM. Error against ground truth: 0.0 BPM.

**Sliding window estimate**: 10-second windows stepped every 1 second produce a time-varying HR trace. Mean absolute error against ground truth HR: 17.1 BPM. Windows that overlap the motion artifact at t=32s produce erroneous estimates. Clean windows (20–35s) track ground truth closely.

The discrepancy between global and windowed performance demonstrates the fundamental tradeoff in rPPG: longer analysis windows average out noise and give stable estimates but sacrifice temporal resolution. Short windows are responsive to HR changes but vulnerable to transient artifacts.

### NB06 — Deep Learning Data Preparation

Prepares a clip-based dataset for 3D CNN training.

Each video frame's forehead patch is resized to 32x32 pixels and normalized to [0,1]. A sliding window of T=30 frames with stride=5 creates overlapping clips, each paired with the corresponding 30-sample window of the bandpass-filtered, normalized GT PPG signal as the training label.

Dataset statistics:

| Split | Clips | Frames | Duration |
|---|---|---|---|
| Train (first 80%) | 243 | 7290 | ~42s |
| Val (last 20%) | 61 | 1830 | ~10s |
| Total | 304 | 9120 | ~52s |

The train/val split is time-based, not random. Random splitting would place clips from the same second in both train and val sets — the model would achieve near-perfect validation by memorizing the training signal shape rather than learning to detect pulse patterns.

### NB07 — 3D CNN Model

Implements and trains a compact PhysNet-inspired architecture for end-to-end rPPG signal estimation from video clips.

**Architecture — PhysNetSmall**:

```
Input:  (B, 3, 30, 32, 32)

Block 1  Conv3D(3→32,  kernel=(1,5,5))  + BN + ReLU
         MaxPool3D(kernel=(1,2,2))
         → (B, 32, 30, 16, 16)

Block 2  Conv3D(32→64, kernel=(3,3,3))  + BN + ReLU
         MaxPool3D(kernel=(1,2,2))
         → (B, 64, 30, 8, 8)

Block 3  Conv3D(64→64, kernel=(3,3,3))  + BN + ReLU
         AdaptiveAvgPool3D((30, 1, 1))
         → (B, 64, 30, 1, 1)

Head     Conv1d(64→1, kernel=1)
         → (B, 30)
```

Block 1 uses a spatial-only kernel `(1,5,5)` to extract skin texture features without temporal mixing. Blocks 2 and 3 use full spatiotemporal kernels `(3,3,3)` to learn how spatial patterns evolve over time. Global average pooling collapses the spatial dimensions, and the 1D convolution head projects 64 temporal feature channels down to the rPPG signal.

**Loss function — Pearson correlation**:

```
loss = 1 - mean( pearson_r(pred, target) )
```

Mean squared error is inappropriate for rPPG because the signal amplitude is arbitrary — only the waveform shape and frequency matter. Pearson correlation loss directly optimizes for temporal correlation between predicted and ground truth waveforms regardless of absolute scale.

**Training configuration**:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR schedule | StepLR (x0.5 every 15 epochs) |
| Batch size | 8 |
| Epochs | 40 |

**Results**:

The model exhibits clear overfitting. Training loss decreases monotonically from 0.75 to 0.04. Validation loss oscillates between 0.30 and 0.92 with no convergent trend, indicating the model is memorizing the training portion without learning generalizable pulse detection.

Validation window Pearson correlations:

| Method | Pearson r (val) |
|---|---|
| CHROM | -0.231 |
| 3D CNN | +0.027 |

The negative CHROM correlation reflects signal polarity ambiguity in the short validation window rather than a failure of the method — CHROM achieves 0.0 BPM global error via FFT, which is phase-invariant. The near-zero DL correlation confirms the model learned nothing useful for unseen data from a single video.

---

## Key Results

| Metric | Value |
|---|---|
| CHROM global HR estimate | 106.7 BPM |
| Ground truth HR | 106.7 BPM |
| Global HR error | 0.0 BPM |
| Sliding window MAE | 17.1 BPM |
| 3D CNN val Pearson r | 0.027 |
| CHROM val Pearson r | -0.231 |

---

## Installation

```bash
git clone https://github.com/yourusername/PulseVision.git
cd PulseVision
pip install -r requirements.txt
```

**requirements.txt**:
```
opencv-python
mediapipe
numpy
scipy
matplotlib
jupyter
torch
torchvision
```

Place the UBFC-rPPG subject folder at `data/subject1/` before running the notebooks.

---

## Running the Pipeline

Run notebooks in order from the `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

Each notebook loads its inputs from `../outputs/` and saves its outputs there. The full pipeline runs in approximately 10–15 minutes on CPU.

---

## Limitations

**Single subject**: The entire pipeline is demonstrated on one 52-second recording. The 3D CNN model cannot generalize to other subjects, lighting conditions, or motion patterns without training on a multi-subject dataset.

**Haar cascade ROI**: The frontal face detector fails on profile views and in poor lighting. Landmark-based detection (MediaPipe Face Mesh) would provide more stable ROI coordinates, particularly for the neonatal use case that motivated this project.

**Motion artifact handling**: The spike at t=32s is visible in the RGB signals and corrupts sliding window HR estimates. A robust pipeline would detect and interpolate across artifact segments before signal processing.

**Fixed bandpass range**: The 0.7–4.0 Hz filter assumes an adult resting-to-active HR range. For neonatal monitoring (normal HR 100–180 BPM = 1.67–3.0 Hz), the filter boundaries should be adjusted accordingly.

---

## What Would Improve DL Performance

1. **More subjects**: Training on all 42 UBFC subjects, or the full NBHR neonatal dataset (257 subjects), would provide the diversity needed for generalization.
2. **Data augmentation**: Random brightness/contrast jitter, temporal flipping, and Gaussian noise injection during training would force the model to learn frequency rather than memorize amplitude patterns.
3. **Deeper architecture**: The full PhysNet architecture or transformer-based methods (PhysFormer, EfficientPhys) have larger capacity and are pretrained on public datasets.
4. **Negative Pearson + MSE combined loss**: Adding a frequency-domain loss term penalizing incorrect HR frequency would directly optimize the downstream metric.

---

## Concepts Covered

**Image processing**: color space conversion, spatial region of interest extraction, face detection with Haar cascades, per-frame pixel statistics, RGB channel decomposition.

**Signal processing**: spatial averaging, detrending, Butterworth bandpass filter design, the CHROM algorithm, Fast Fourier Transform, frequency-domain peak detection, sliding window analysis, Pearson correlation.

**Deep learning**: 3D convolutional neural networks, spatiotemporal feature extraction, custom loss functions, training loop design, overfitting diagnosis, train/val split strategy for time-series data.

**Biomedical signal analysis**: photoplethysmography principles, light-tissue interaction and hemoglobin absorption spectra, motion artifact identification, HR estimation from noisy biosignals.

---

## References

- de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886. — CHROM algorithm.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445. — foundational rPPG paper.
- Huang, B., et al. (2021). A neonatal dataset and benchmark for non-contact neonatal heart rate monitoring based on spatio-temporal neural networks. *Engineering Applications of Artificial Intelligence*, 106, 104447. — NBHR dataset.
- Bobbia, S., et al. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90. — UBFC-rPPG dataset.
- Yu, Z., et al. (2019). Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks. *British Machine Vision Conference*. — PhysNet architecture.
