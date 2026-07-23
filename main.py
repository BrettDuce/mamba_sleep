import os
# Optimise CUDA memory allocation via PyTorch memory management
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
import matplotlib
from scipy.ndimage import label
matplotlib.use('Agg')

# Project modular imports
from sleep_model import SleepStagingBaseline3Ch
from arousal_model import ArousalSegmentationModel4Ch, DurationPenalizedFocalTverskyLoss 
from train_eval import (
    evaluate_models, 
    plot_confusion_matrix, 
    plot_hypnogram_comparison, 
    plot_hypnodensity, 
    plot_transition_matrix, 
    plot_arousal_confusion_matrix,
    plot_event_raster
)

# =====================================================================
# SYSTEM RUN CONFIGURATION
# =====================================================================

CONFIG = {
    'n_channels': 4,            # Input signal channels (EEG, EOG, EMG, ECG)
    'seq_len': 100,             # Number of 30-second epochs per sequence batch
    'batch_size': 1,            # Batch size (1 whole PSG sequence at a time)
    'epochs': 150,              # Maximum training iteration cap
    'lr': 5e-5,                 # Learning rate for AdamW optimiser
    'phase': 1,                 # 1: Sleep Stage Training, 2: Arousal Segmentation Training
    'wake_trim_epochs': 60      # Number of 30s wake epochs retained before/after sleep
}


# =====================================================================
# CLINICAL METRICS AND METADATA GENERATORS
# =====================================================================

def calculate_architecture_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates primary clinical sleep architecture indices from ground-truth and predicted labels.

    Args:
        y_true (np.ndarray): Target sleep stage labels. Shape: (num_epochs,)
        y_pred (np.ndarray): Predicted sleep stage labels. Shape: (num_epochs,)

    Returns:
        dict: Pairs of (ground_truth, prediction) for TST, SOL, WASO, and SE.
    """
    valid_mask = y_true != -100
    yt, yp = y_true[valid_mask], y_pred[valid_mask]
    total_time_min = len(yt) * 0.5  # Each epoch represents 0.5 minutes (30s)

    def get_sol(labels):
        sleep_indices = np.where(labels > 0)[0]  # First non-wake epoch
        return sleep_indices[0] * 0.5 if len(sleep_indices) > 0 else total_time_min

    def get_waso(labels):
        sleep_indices = np.where(labels > 0)[0]
        if len(sleep_indices) < 2: 
            return 0.0
        onset, offset = sleep_indices[0], sleep_indices[-1]
        return np.sum(labels[onset:offset] == 0) * 0.5

    tst_true, tst_pred = np.sum(yt > 0) * 0.5, np.sum(yp > 0) * 0.5
    sol_true, sol_pred = get_sol(yt), get_sol(yp)
    waso_true, waso_pred = get_waso(yt), get_waso(yp)
    se_true, se_pred = (tst_true / total_time_min) * 100.0, (tst_pred / total_time_min) * 100.0

    return {
        'TST': (tst_true, tst_pred),    # Total Sleep Time (minutes)
        'SOL': (sol_true, sol_pred),    # Sleep Onset Latency (minutes)
        'WASO': (waso_true, waso_pred), # Wake After Sleep Onset (minutes)
        'SE': (se_true, se_pred)        # Sleep Efficiency (%)
    }


def generate_metadata(feature_path: str):
    """
    Scans processed PyTorch data files to profile event densities (arousals and Stage N1)
    and exports a JSON dataset inventory.

    Args:
        feature_path (str): Directory containing preprocessed .pt files.
    """
    files = glob.glob(os.path.join(feature_path, "*.pt"))
    metadata = {}
    
    for f in tqdm(files, desc="Scanning for Arousals and N1"):
        try:
            data = torch.load(f, map_location='cpu')
            y_a = data['y_arousal']  # Shape: (num_epochs, samples_per_epoch)
            y_s = data['y_stage']    # Shape: (num_epochs,)
            
            arousal_density = (y_a > 0.5).float().mean().item()
            n1_density = (y_s == 1).float().mean().item()
            
            metadata[os.path.basename(f)] = {
                "a_density": arousal_density,
                "n1_density": n1_density,
                "has_events": arousal_density > 0 or n1_density > 0
            }
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
        
    with open("dataset_metadata.json", "w") as j:
        json.dump(metadata, j)


# =====================================================================
# DATASET & SAMPLING ENGINE
# =====================================================================

class PolysomnographyDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset engine for reading multi-channel polysomnography (PSG) records,
    handling wake-trimming bounds, dynamic sequence cropping, and padding.
    """
    def __init__(self, file_list: list, seq_len: int = 100, is_training: bool = True, trim_wake: bool = False): 
        self.file_list = file_list
        self.seq_len = seq_len
        self.is_training = is_training
        self.trim_wake = trim_wake
        
        self.boundaries = []
        if self.trim_wake:
            print("Calculating Wake-Trim boundaries...")
            for f in tqdm(file_list, desc="Trimming"):
                data = torch.load(f, map_location='cpu')
                y = data['y_stage'].numpy()
                sleep_idx = np.where(y > 0)[0]
                
                if len(sleep_idx) > 0:
                    # Retain a fixed buffer of wake epochs prior to sleep onset and after sleep offset
                    start = max(0, sleep_idx[0] - CONFIG['wake_trim_epochs'])
                    end = min(len(y), sleep_idx[-1] + CONFIG['wake_trim_epochs'] + 1)
                    self.boundaries.append((start, end))
                else:
                    self.boundaries.append((0, len(y)))
        else:
            self.boundaries = [(0, None)] * len(file_list)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        """
        Loads a single PSG record.

        Returns:
            x (torch.Tensor): Signal tensor. Shape: (seq_len, n_channels, samples_per_epoch)
            y (torch.Tensor): Stage label tensor. Shape: (seq_len,)
            y_a (torch.Tensor): Arousal mask tensor. Shape: (seq_len, samples_per_epoch)
        """
        data = torch.load(self.file_list[idx], map_location='cpu')
        start_trim, end_trim = self.boundaries[idx]
        
        x = data['x'][start_trim:end_trim]        # Target layout: (epochs, channels, samples)
        y = data['y_stage'][start_trim:end_trim]
        y_a = data['y_arousal'][start_trim:end_trim]
        
        saved_channels = x.shape[1]
        if saved_channels != CONFIG['n_channels']:
            raise ValueError(
                f"Tensor dimension failure inside {self.file_list[idx]}. "
                f"Expected {CONFIG['n_channels']} channels at shape index 1, but found {saved_channels}. "
                f"File matrix shape layout is: {list(x.shape)}"
            )
        
        n_epochs = x.shape[0]
        # Crop or pad sequences to maintain uniform seq_len dimension
        if n_epochs > self.seq_len:
            start = np.random.randint(0, n_epochs - self.seq_len) if self.is_training else 0
            x = x[start : start + self.seq_len]
            y = y[start : start + self.seq_len]
            y_a = y_a[start : start + self.seq_len]
        elif n_epochs < self.seq_len:
            pad_len = self.seq_len - n_epochs
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            y = F.pad(y, (0, pad_len), value=0) 
            y_a = F.pad(y_a, (0, 0, 0, pad_len))
            
        return x.float(), y.long(), y_a.float()

    def get_labels(self) -> np.ndarray:
        """Extracts dominant class labels per file to configure the WeightedRandomSampler."""
        print("[INFO] Calculating dataset weights for WeightedRandomSampler...")
        labels = []
        for idx in range(len(self.file_list)):
            data = torch.load(self.file_list[idx], map_location='cpu')
            start_trim, end_trim = self.boundaries[idx]
            y = data['y_stage'][start_trim:end_trim].numpy()
            
            unique, counts = np.unique(y, return_counts=True)
            stage_counts = dict(zip(unique, counts))
            total_epochs = len(y) if len(y) > 0 else 1
            
            n1_ratio = stage_counts.get(1, 0) / total_epochs
            
            # Prioritise minority Stage N1 representation
            if n1_ratio > 0.05:
                labels.append(1)
            else:
                labels.append(int(unique[np.argmax(counts)]))
                
        return np.array(labels)


# =====================================================================
# LOSS FUNCTIONS & TRAINING LOOPS
# =====================================================================

class MulticlassFocalLoss(nn.Module):
    """Focal Loss with label smoothing for handling severe sleep stage class imbalances."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Inputs shape: (N, C), Targets shape: (N,)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', 
            weight=self.weight, label_smoothing=self.smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_one_epoch(sleep_model: nn.Module, arousal_model: nn.Module, loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, 
                    sleep_crit: nn.Module, arousal_crit: nn.Module, phase: int, 
                    accumulation_steps: int = 16) -> float:
    """
    Executes a single training epoch across either Phase 1 (Staging) or Phase 2 (Arousal Segmentation).

    Args:
        sleep_model (nn.Module): Sleep staging model.
        arousal_model (nn.Module): Arousal segmentation model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimiser instance.
        scaler (GradScaler): Automatic Mixed Precision (AMP) gradient scaler.
        device (torch.device): CUDA compute target.
        sleep_crit (nn.Module): Loss function for staging.
        arousal_crit (nn.Module): Loss function for segmentation.
        phase (int): Active training phase (1 or 2).
        accumulation_steps (int): Gradient accumulation frequency.

    Returns:
        float: Mean epoch loss value.
    """
    if phase == 1:
        sleep_model.train()
        desc = "Phase 1: Training Staging Baseline"
    elif phase == 2:
        sleep_model.eval() 
        arousal_model.train()
        desc = "Phase 2: Training Decoupled Arousal Pipeline"
        
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    for i, (x, y_s, y_a) in enumerate(tqdm(loader, desc=desc)):
        # x shape: (B, S, 4, 3000), y_s shape: (B, S), y_a shape: (B, S, 60)
        x, y_s, y_a = x.to(device), y_s.to(device), y_a.to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if phase == 1:
                # PHASE 1: STAGING OPTIMISATION (3-channel slice from 4-channel raw input)
                b, s, c, t = x.shape
                x_flat = x.view(b * s, c, t)            # Flatten to: (B*S, 4, 3000)
                x_staging = x_flat[:, :3, :]            # Slice channels: (B*S, 3, 3000)
                
                logits_s = sleep_model(x_staging)      # Output shape: (B*S, 5)
                loss = sleep_crit(logits_s.view(-1, 5), y_s.view(-1))
                
            elif phase == 2:
                # PHASE 2: AROUSAL SEGMENTATION OPTIMISATION
                with torch.no_grad():
                    b, s, c, t = x.shape
                    x_staging = x[:, :, :3, :].view(b * s, 3, t)
                    logits_s = sleep_model(x_staging)
                    stage_probs = torch.softmax(logits_s, dim=-1).view(b, s, 5)  # Context shape: (B, S, 5)
                
                # Pass full 4-channel tensor and context probabilities to arousal model
                logits_a = arousal_model(x, stage_probs) # Output shape: (B, S, 60)
                loss = arousal_crit(logits_a, y_a)

        # Scale loss and accumulate gradients
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            
            # Clip gradient norms to prevent exploding gradients
            if phase == 1:
                torch.nn.utils.clip_grad_norm_(sleep_model.parameters(), 1.0)
            elif phase == 2:
                torch.nn.utils.clip_grad_norm_(arousal_model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        total_loss += loss.item() * accumulation_steps
        
    return total_loss / len(loader)


# =====================================================================
# MAIN PIPELINE ORCHESTRATOR
# =====================================================================

def main():
    overall_start = datetime.now()
    device = torch.device("cuda")
    phase = CONFIG['phase'] 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"run_Phase{phase}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    if phase == 1:
        print("INITIALISING PHASE 1: STAGING BASELINE ANALYSIS")
    elif phase == 2:
        print("INITIALISING PHASE 2: DECOUPLED AROUSAL SEGMENTATION PIPELINE")
    print(f"Wake Trim Active: Keeping {CONFIG['wake_trim_epochs']} epochs of buffer.")
    print(f"Run Directory: {run_dir}")
    print("=" * 50 + "\n")

    # Locate dataset records
    data_files = sorted(glob.glob(os.path.join(r"C:\Users\brett\shhs_features_4ch_128hz", "*.pt")))
    
    # Configure deterministic splits
    train_val_files, test_files = train_test_split(data_files, test_size=0.10, random_state=42)
    train_f, val_f = train_test_split(train_val_files, test_size=0.1111, random_state=42)

    print(f"Dataset Split: {len(train_f)} Train | {len(val_f)} Validate | {len(test_files)} Test")

    trim_setting = True 

    train_dataset = PolysomnographyDataset(train_f, is_training=True, trim_wake=trim_setting)
    val_dataset = PolysomnographyDataset(val_f, is_training=False, trim_wake=trim_setting)

    # Configure class-weighted sampler
    train_labels = train_dataset.get_labels()
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1.0 / (class_sample_count + 1e-5)
    samples_weight = torch.from_numpy(np.array([weight[np.where(np.unique(train_labels) == t)[0][0]] for t in train_labels])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Model initialisation
    sleep_model = SleepStagingBaseline3Ch(n_channels=3, n_classes=5).to(device) 
    arousal_model = None

    if phase == 1:
        print("\n[INFO] INITIALISING PHASE 1: TRAINING STAGING BASELINE")
        optimizer = torch.optim.AdamW(sleep_model.parameters(), lr=CONFIG['lr'], weight_decay=5e-4)

    elif phase == 2:
        print("\n[INFO] INITIALISING PHASE 2: TRAINING 4-CHANNEL AROUSAL SEGMENTATION MODEL")
        arousal_model = ArousalSegmentationModel4Ch(n_channels=4, n_classes=5).to(device)
        
        try:
            sleep_model.load_state_dict(torch.load("best_sleep_staging_3ch.pt", map_location=device))
            print("[SUCCESS] Loaded Pre-trained Sleep Staging Model Checkpoint.")
            # Freeze staging weights during Phase 2
            for param in sleep_model.parameters():
                param.requires_grad = False
            print("[INFO] Staging Model Parameters Frozen.")
        except Exception as e:
            print(f"[ERROR] Cannot execute Phase 2 without a pre-trained staging baseline weight profile! ({e})")
            return
            
        optimizer = torch.optim.AdamW(arousal_model.parameters(), lr=CONFIG['lr'], weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    # Loss criteria definitions
    weights = torch.tensor([1.2, 2.5, 1.1, 1.7, 1.1], dtype=torch.float).to(device)
    sleep_crit = MulticlassFocalLoss(gamma=2.0, weight=weights, smoothing=0.1)
    arousal_crit = DurationPenalizedFocalTverskyLoss(
        alpha=0.3, 
        beta=0.7, 
        gamma=2.0, 
        roi_expansion=0, 
        max_duration_sec=15.0, 
        penalty_weight=5.0
    )

    # Restore historical performance benchmarks
    try:
        with open("pipeline_performance_metrics.json", "r") as f:
            records = json.load(f)
            global_best_kappa = records.get("best_kappa", 0.0)
            global_best_macro_f1 = records.get("best_macro_f1", 0.0)
            global_best_arousal_f1 = records.get("best_arousal_f1", 0.0)
    except FileNotFoundError:
        global_best_kappa, global_best_macro_f1, global_best_arousal_f1 = 0.0, 0.0, 0.0
    
    start_epoch = 0
    resume_checkpoint = "" 
    
    if os.path.exists(resume_checkpoint):
        print(f"[RECOVERY] Restoring {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if phase == 1:
            sleep_model.load_state_dict(checkpoint['sleep_state_dict'])
        elif phase == 2:
            arousal_model.load_state_dict(checkpoint['arousal_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        print(f"Fast-forwarding training directly to Epoch {start_epoch}")

    print(f"Training start time: {overall_start.strftime('%H:%M:%S')}")

    # --- MAIN OPTIMISATION LOOP ---
    for epoch in range(start_epoch, CONFIG['epochs']):

        train_loss = train_one_epoch(
            sleep_model, arousal_model, train_loader, optimizer, scaler, 
            device, sleep_crit, arousal_crit, phase=phase
        )
        scheduler.step()

        print(f"Epoch {epoch + 1:03d} | Total Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Periodic evaluation and model checkpointing
        if (epoch + 1) % 5 == 0:
            results, _, metrics = evaluate_models(
                sleep_model, arousal_model, val_loader, 
                device, phase=phase, epoch=epoch, save_path=run_dir
            )
            
            current_kappa = metrics.get('kappa', 0.0)
            current_arousal_f1 = metrics.get('arousal_f1', 0.0)
            
            if phase == 1 and current_kappa > global_best_kappa:
                global_best_kappa = current_kappa
                torch.save(sleep_model.state_dict(), "best_sleep_staging_3ch.pt")
                print(f"NEW GLOBAL BEST STAGING! Kappa: {global_best_kappa:.4f}")
                
            if phase == 2 and current_arousal_f1 > global_best_arousal_f1:
                global_best_arousal_f1 = current_arousal_f1
                torch.save(arousal_model.state_dict(), "best_arousal_segmentation_5ch.pt")
                print(f"New Global Best Arousal F1: {global_best_arousal_f1:.4f}")
                
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'sleep_state_dict': sleep_model.state_dict() if phase == 1 else None,
                    'arousal_state_dict': arousal_model.state_dict() if phase == 2 else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'phase': phase
                }
                torch.save(checkpoint, os.path.join(run_dir, f"checkpoint_ep{epoch + 1}.pt"))
                print(f"Checkpoint saved at Epoch {epoch + 1}")
            
            with open("pipeline_performance_metrics.json", "w") as f:
                json.dump({
                    "best_kappa": global_best_kappa,
                    "best_macro_f1": metrics.get('macro_f1', global_best_macro_f1),
                    "best_arousal_f1": global_best_arousal_f1
                }, f)

    # Save final model weights
    final_path = os.path.join(run_dir, f"phase{phase}_final_epoch{CONFIG['epochs']}.pt")
    if phase == 1:
        torch.save(sleep_model.state_dict(), final_path)
    elif phase == 2:
        torch.save(arousal_model.state_dict(), final_path)
        
    overall_end = datetime.now()
    total_duration = overall_end - overall_start
    duration_str = str(total_duration).split('.')[0] 

    print("\n" + "=" * 50)
    print(f"PHASE {phase} TRAINING ROUTINE COMPLETE. GO HAVE A BEER!")
    print(f"Total Time Elapsed: {duration_str}")
    print(f"Final models and plots are in: {run_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
