# CS 441 Final Project – Audio-Based Tennis Impact Localization

This project estimates the impact location of a tennis ball on a racket face using **audio signals only**.  
By analyzing the sound produced during ball–racket contact, the system predicts the normalized 2D impact location on a canonical racket template, without relying on high-speed cameras or instrumented rackets.

---

## Project Overview

The core idea of this project is that the **transient acoustic pattern** generated at impact encodes information about where the ball hits the racket.  
I design an end-to-end pipeline that extracts and aligns impact audio events, constructs training labels from annotated hit locations, and evaluates multiple regression models to map audio signals to spatial coordinates.

---

## Data Processing Pipeline

1. **Video & Audio Extraction**
   - `get_frames.py`: Extracts video frames from recorded videos.
   - `extract_audio_and_plot.py`: Extracts audio tracks and visualizes waveforms.

2. **Impact Event Selection**
   - `apply_mask_via_keypoints.py`: Uses keypoint-based masking to isolate the racket region.
   - `link_audio_to_good_frames.py`: Links valid impact frames to corresponding audio segments.

3. **Label Construction**
   - Impact locations are normalized using a racket template.
   - Labels are stored as normalized Cartesian coordinates `(x_norm, y_norm)` or normalized polar coordinates.

---

## Models and Methods

### Tree-Based Regression on Raw Waveform
- **Random Forest** (`rf_audio_to_coord.py`)
- **Extra Trees** (`extratrees_raw_waveform.py`)

Both models use:
- RMS energy peak detection to align the waveform to the impact moment
- Fixed-length raw waveform segments (~120 ms)
- Amplitude normalization to reduce recording variability

Extra Trees introduces stronger randomness during tree splitting and achieved the best overall performance.

### Neural Network Baseline
- **1D CNN**
  - Takes peak-aligned raw waveforms as input
  - Learns local temporal patterns using convolution and pooling
  - More expressive but more sensitive to noise and data size

---

## Evaluation Metrics

Models are evaluated using:
- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **Average radial error** in normalized racket coordinates

These metrics reflect both numerical accuracy and spatial error on the racket face.

---

## Repository Structure

