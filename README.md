# MambaSleep

PyTorch implementation of MambaSleep, a selective state space architecture designed for joint automated sleep staging using 1 EEG, 1 EOG, and 1 EMG channel and arousal detection using 1 EEG, 1 EOG, 1 EMG and the ECG channel.

## Overview

MambaSleep leverages selective state space models (S4/Mamba) to process multi-channel polysomnography (PSG) data. The architecture captures long-range temporal dependencies required for sequence-to-sequence sleep scoring and transient micro-arousal identification.

This repository includes data parsing for Compumedics Profusion study annotations, feature preprocessing, model definitions, and evaluation pipelines.

## Key Features

Selective State Space Architecture: High-efficiency sequence modeling optimized for continuous PSG signals.

Dual Task Support: Pipelines for both 5-stage sleep classification (AASM rules) and continuous cortical arousal detection.

Standardised 3-Channel Input: Built for 1 EEG, 1 EOG, and 1 EMG channel configurations.

Compumedics Profusion Parser: Automated extraction and alignment of signal data and XML annotation files.

Evaluation Metrics: Standardized reporting for epoch-level accuracy, macro F1, Cohen's kappa (staging), and AUPRC/AUROC (arousals).

## Installation Prerequisites

Python 3.10+

CUDA-compatible GPU

PyTorch 2.0+


## Setup
Clone the repository:

```Bash
git clone [https://github.com/your-username/mamba_sleep.git](https://github.com/your-username/mamba_sleep.git)
cd mamba_sleep
```

Create and activate a virtual environment:

```Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required dependencies:

```Bash
pip install -r requirements.txt
```
Note: Ensure causal-conv1d and mamba-ssm are installed correctly for your specific CUDA version.

## Usage

1. Data Preparation

Prepare your raw EDF files into preprocessed NumPy array formats:

```Bash
python scripts/preprocess.py --data_dir /path/to/raw_edf/ --output_dir ./processed_data/
```

2. Training

Train the model using a configuration file:

```Bash
python scripts/train.py --config configs/mamba_sleep_base.yaml
```

3. Evaluation

Run evaluation on a pre-trained checkpoint:

```Bash
python scripts/evaluate.py --checkpoint /path/to/model.pt --test_data ./processed_data/test/
```

## Citation
If you use this codebase in your research, please cite:

Code snippet
@article{duce2026mambasleep,
  title={MambaSleep: Automated Sleep Staging via Selective State Space Models},
  author={Duce, Brett},
  journal={},
  year={2026}
}
License
This project is licensed under the MIT License. See the LICENSE file for details.
