import torch
import matplotlib.pyplot as plt
import os
import glob
import random

# =====================================================================
# DATASET TENSOR VERIFICATION & VISUALISATION ENGINE
# =====================================================================

def verify_pt_file(file_path: str):
    """
    Loads, inspects, and validates preprocessed PyTorch dataset files (.pt).
    
    Verifies dictionary integrity, tensor shape dimensions, data types, 
    channel normalisation statistics (mean/std), presence of NaNs, and 
    plots multi-channel signal waveforms alongside 10 Hz arousal masks.

    Args:
        file_path (str): File path to the target .pt dataset file.
    """
    print(f"--- Checking: {os.path.basename(file_path)} ---")
    
    # 1. Load data dictionary
    try:
        # Set weights_only=False to allow loading custom dictionary objects containing PyTorch tensors
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    # 2. Structure Safety Verification: Ensure the loaded object is a dictionary
    if not isinstance(data, dict):
        print(f"⚠️ ERROR: This file is a raw {type(data)}, not a dictionary!")
        print("This means the file was saved incorrectly. Re-run your ingestion script.")
        if torch.is_tensor(data):
            print(f"Tensor Shape: {data.shape}")
        return

    # 3. Keys and Dimension Inspection
    expected_keys = ['x', 'y_stage', 'y_arousal']
    all_keys_present = True
    
    for key in expected_keys:
        if key in data:
            print(f"✅ Key found: {key:10} | Shape: {str(data[key].shape):20} | Dtype: {data[key].dtype}")
        else:
            print(f"❌ Missing key: {key}")
            all_keys_present = False

    if not all_keys_present:
        return

    # Extract tensors and document expected shapes
    x = data['x']           # Signal shape: (Num_Epochs, Channels=3, Signal_Samples=3000)
    y_s = data['y_stage']   # Stage shape: (Num_Epochs,)
    y_a = data['y_arousal'] # Arousal mask shape: (Num_Epochs, Mask_Samples=300)

    # 4. Numerical Sanity & Normalisation Checks
    mean_val = x.mean().item()
    std_val = x.std().item()
    has_nan = torch.isnan(x).any().item()
    
    print("\n--- Statistics ---")
    print(f"Mean: {mean_val:7.4f} (Target: ~0.0)")
    print(f"Std:  {std_val:7.4f} (Target: ~1.0)")
    print(f"NaNs: {'⚠️ YES!' if has_nan else '✅ None'}")
    
    # 5. Visualisation Pipeline
    # Prioritise selecting an epoch containing an active arousal event for visual confirmation
    arousal_indices = (y_a.sum(dim=1) > 0).nonzero()
    
    if len(arousal_indices) > 0:
        epoch_idx = arousal_indices[random.randint(0, len(arousal_indices) - 1)].item()
        print(f"\nPlotting Epoch {epoch_idx} (Selected because it contains an arousal event)")
    else:
        epoch_idx = random.randint(0, x.shape[0] - 1)
        print(f"\nPlotting Random Epoch {epoch_idx}")

    print(f"Sleep Stage Code: {y_s[epoch_idx].item()}")

    # Render channel traces and binary arousal overlay
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    channels = ['EEG C4-M1', 'EOG E2-M1', 'EMG Chin']
    
    # Plot electrophysiological signal waveforms (3000 samples at 100 Hz = 30 seconds)
    for i in range(3):
        axes[i].plot(x[epoch_idx, i].numpy(), lw=0.7, color='darkblue')
        axes[i].set_ylabel(channels[i])
        axes[i].grid(True, alpha=0.2)

    # Upsample 300-point mask (10 Hz) by a factor of 10 to align visually with 3000-point signal (100 Hz)
    mask_upsampled = torch.repeat_interleave(y_a[epoch_idx], 10).numpy() # Resampled shape: (3000,)
    
    axes[3].fill_between(range(3000), 0, mask_upsampled, color='red', alpha=0.4, label='Arousal')
    axes[3].set_ylabel("Arousal Mask")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend(loc='upper right')

    plt.suptitle(f"Data Check: {os.path.basename(file_path)}")
    plt.tight_layout()
    plt.show()


# =====================================================================
# BATCH EXECUTION ORCHESTRATOR
# =====================================================================

if __name__ == "__main__":
    FEATURE_PATH = r"F:\Data\Features"  ### CHANGE THE FEATURE PATH FOR YOUR OWN PURPOSES ###
    files = glob.glob(os.path.join(FEATURE_PATH, "*.pt"))
    
    if not files:
        print(f"No files found in {FEATURE_PATH}!")
    else:
        # Select and verify a random record from the feature folder
        verify_pt_file(random.choice(files))
