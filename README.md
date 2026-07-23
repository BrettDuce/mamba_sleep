# mamba_sleep
PyTorch implementation of MambaSleep, a selective state space architecture for automated sleep staging using 1 EEG, 1 EOG, and 1 EMG channel. Includes preprocessing scripts, model architectures based on selective state spaces, and evaluation tools for standard benchmark datasets. XML parses is for Compumedics Profusion studies.

Installation Prerequisites

Python 3.10+

CUDA-compatible GPU

PyTorch 2.0+


Setup
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

Usage

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

Citation
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
