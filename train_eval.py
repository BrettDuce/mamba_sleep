import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy.ndimage import label, binary_dilation, binary_erosion, median_filter
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score, 
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, precision_score, f1_score, precision_recall_curve, auc,
    precision_recall_fscore_support, recall_score
)

# =====================================================================
# 1. CLINICAL RULES & HELPERS
# =====================================================================

def filter_duration(mask: np.ndarray, min_sec: float = 3.0, hz: int = 2) -> np.ndarray:
    """
    Removes contiguous positive events in a binary mask that fall below a minimum duration.

    Args:
        mask (np.ndarray): 1D binary array of predictions (1s and 0s).
        min_sec (float): Minimum event duration threshold in seconds. Default is 3.0s.
        hz (int): Sampling frequency of the input mask array in Hz. Default is 2Hz.

    Returns:
        np.ndarray: Filtered binary mask with sub-threshold events zeroed out.
    """
    labeled_array, num_features = label(mask)
    new_mask = np.zeros_like(mask)
    min_frames = int(min_sec * hz)  # Convert seconds to sample frames (e.g., 3s * 2Hz = 6 frames)
    
    for i in range(1, num_features + 1):
        slice_x = np.where(labeled_array == i)[0]
        # Retain only events with enough frames to meet the clinical duration threshold
        if len(slice_x) >= min_frames:
            new_mask[slice_x] = 1
            
    return new_mask


def connect_events(mask: np.ndarray, gap_sec: float = 1.5, hz: int = 2) -> np.ndarray:
    """
    Bridges small temporal gaps between adjacent positive events using morphological closing.

    Args:
        mask (np.ndarray or torch.Tensor): Input binary event mask.
        gap_sec (float): Maximum gap duration in seconds to bridge. Default is 1.5s.
        hz (int): Sampling frequency of the input mask in Hz. Default is 2Hz.

    Returns:
        np.ndarray: Binary mask with closely spaced events merged.
    """
    if hasattr(mask, 'cpu'): 
        mask = mask.cpu().numpy()
        
    gap_frames = int((gap_sec * hz) // 2)  # Radius footprint for morphological closing
    if gap_frames < 1: 
        return mask
        
    # Morphological closing: Dilation expands boundaries, Erosion shrinks them back
    dilated = binary_dilation(mask, iterations=gap_frames)
    connected = binary_erosion(dilated, iterations=gap_frames)
    return connected.astype(np.int8)


def clinical_arousal_filter(mask: np.ndarray, hz: int = 2) -> np.ndarray:
    """
    Applies standard clinical scoring rules to raw arousal predictions.
    Bridges short gaps (<1.5s) and enforces minimum (3s) and maximum (15s) event duration constraints.

    Args:
        mask (np.ndarray or torch.Tensor): Raw binary prediction array.
        hz (int): Signal array frequency in Hz. Default is 2Hz.

    Returns:
        np.ndarray: Clinically post-processed binary arousal mask.
    """
    if hasattr(mask, 'cpu'): 
        mask = mask.cpu().numpy()
    
    # 1. Connect Gaps: Bridge 1.5s gaps (1 iteration at 2Hz covers 1.5s window)
    gap_frames = 1 
    dilated = binary_dilation(mask, iterations=gap_frames)
    mask = binary_erosion(dilated, iterations=gap_frames)

    # 2. Enforce Clinical Duration Rules (3s to 15s)
    labeled_array, num_features = label(mask)
    for i in range(1, num_features + 1):
        event_length = np.sum(labeled_array == i)
        
        # At 2Hz: 3 seconds = 6 samples | 15 seconds = 30 samples
        # Delete events shorter than 3s or longer than 15s
        if event_length < 6 or event_length > 30:
            mask[labeled_array == i] = 0
            
    return mask.astype(np.int8)


def apply_rem_emg_rule(arousal_mask: np.ndarray, stage_mask: np.ndarray, 
                       chin_emg_100hz: np.ndarray, hz: int = 2) -> np.ndarray:
    """
    Refines REM sleep arousal candidate events by requiring a concurrent increase in submental EMG 
    amplitude relative to the preceding 10-second baseline (standard AASM clinical requirement).

    Args:
        arousal_mask (np.ndarray): 2Hz predicted arousal mask. Shape: (num_samples_2hz,)
        stage_mask (np.ndarray): Sleep stage epoch array mapped to samples. Shape: (num_samples_2hz,)
        chin_emg_100hz (np.ndarray): High-resolution 100Hz raw submental EMG signal.
        hz (int): Base sampling frequency of the arousal mask. Default is 2Hz.

    Returns:
        np.ndarray: Refined arousal mask with unconfirmed REM candidate events removed.
    """
    refined_mask = arousal_mask.copy()
    labeled_arousals, num_events = label(arousal_mask)
    
    for i in range(1, num_events + 1):
        idx = np.where(labeled_arousals == i)[0]
        
        # Stage 4 represents REM sleep in this mapping
        if np.any(stage_mask[idx] == 4): 
            # Map 2Hz index pointers to high-resolution 100Hz raw EMG positions (1 frame at 2Hz = 50 samples at 100Hz)
            start_100, end_100 = idx[0] * 50, idx[-1] * 50
            
            # Extract 10-second lookback baseline window (1000 samples at 100Hz)
            base_start = max(0, start_100 - 1000) 
            base_var = np.var(chin_emg_100hz[base_start:start_100]) + 1e-6
            event_var = np.var(chin_emg_100hz[start_100:end_100])
            
            # Suppress REM arousal if EMG variance does not double compared to baseline
            if event_var < (2.0 * base_var): 
                refined_mask[idx] = 0
                
    return refined_mask


def apply_clinical_respiratory_rules(pred_mask: np.ndarray, stage_mask: np.ndarray, hz: int = 2) -> np.ndarray:
    """
    Enforces clinical duration rules on respiratory events (e.g., minimum 10-second duration during sleep).

    Args:
        pred_mask (np.ndarray): Raw prediction mask. Shape: (num_samples_2hz,)
        stage_mask (np.ndarray): Sleep stage labels across time. Shape: (num_samples_2hz,)
        hz (int): Prediction resolution in Hz. Default is 2Hz.

    Returns:
        np.ndarray: Filtered respiratory event mask.
    """
    min_frames = 10 * hz  # 10-second rule equals 20 frames at 2Hz
    labeled_mask, num_features = label(pred_mask)
    clinical_mask = np.zeros_like(pred_mask)
    
    for i in range(1, num_features + 1):
        seg = (labeled_mask == i)
        # Event must meet 10s length AND occur during active sleep (stage > 0)
        if seg.sum() >= min_frames and np.any(stage_mask[seg] > 0):
            clinical_mask[seg] = 1
            
    return clinical_mask


def calculate_event_iou_metrics(y_true: np.ndarray, y_pred: np.ndarray, iou_threshold: float = 0.3):
    """
    Calculates event-level Precision, Recall, and F1-score using an Intersection-over-Union (IoU) metric.

    Args:
        y_true (np.ndarray): Ground truth event mask. Shape: (N,)
        y_pred (np.ndarray): Predicted event mask. Shape: (N,)
        iou_threshold (float): Minimum IoU overlap needed to score a True Positive. Default is 0.3.

    Returns:
        tuple: (precision, recall, f1_score, tp_count, fp_count, fn_count)
    """
    def get_events(mask):
        labeled, n = label(mask)
        return [np.where(labeled == i)[0] for i in range(1, n + 1)]

    true_events = get_events(y_true)
    pred_events = get_events(y_pred)
    
    tp = 0
    matched_true = set()
    
    for p_ev in pred_events:
        for t_idx, t_ev in enumerate(true_events):
            if t_idx in matched_true: 
                continue
                
            intersection = len(np.intersect1d(p_ev, t_ev))
            union = len(np.union1d(p_ev, t_ev))
            
            if (intersection / (union + 1e-6)) >= iou_threshold:
                tp += 1
                matched_true.add(t_idx)
                break
                
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return prec, rec, f1, tp, fp, fn


def sweep_n3_boundary(y_true: np.ndarray, s_probs: np.ndarray) -> float:
    """
    Performs a grid search sweep across probability thresholds for upgrading N2 predictions to N3.
    Optimises the decision boundary between Stage N2 and Stage N3 deep sleep.

    Args:
        y_true (np.ndarray): Ground truth stage labels. Shape: (n_epochs,)
        s_probs (np.ndarray): Model output softmax probabilities. Shape: (n_epochs, 5)

    Returns:
        float: Optimal probability threshold yielding the highest N3 F1-score.
    """
    thresholds = np.arange(0.20, 0.51, 0.02)
    N2_IDX, N3_IDX = 2, 3
    
    print("\n" + "=" * 50)
    print("N3 DECISION BOUNDARY SWEEP".center(50))
    print("=" * 50)
    print(f"{'Threshold':<10} | {'N3 F1':<10} | {'N2 F1':<10} | {'Overall Kappa':<15}")
    print("-" * 50)
    
    best_n3_f1 = -1.0
    best_thresh = 0.50
    n3_probabilities = s_probs[:, N3_IDX]
    
    for thresh in thresholds:
        adjusted_preds = np.argmax(s_probs, axis=-1)
        
        # Upgrade N2 predictions to N3 if N3 probability crosses custom threshold
        upgrade_mask = (adjusted_preds == N2_IDX) & (n3_probabilities >= thresh)
        adjusted_preds[upgrade_mask] = N3_IDX
        
        f1_scores = f1_score(y_true, adjusted_preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
        kappa = cohen_kappa_score(y_true, adjusted_preds)
        
        n2_f1, n3_f1 = f1_scores[N2_IDX], f1_scores[N3_IDX]
        
        print(f"{thresh:<10.2f} | {n3_f1:<10.4f} | {n2_f1:<10.4f} | {kappa:<15.4f}")
        
        if n3_f1 > best_n3_f1:
            best_n3_f1 = n3_f1
            best_thresh = thresh
            
    print("=" * 50)
    print(f"Optimal N3 Threshold: {best_thresh:.2f} (N3 F1: {best_n3_f1:.4f})\n")
    
    return best_thresh


# =====================================================================
# 2. EVALUATION METRICS ENGINE
# =====================================================================

def evaluate_models(sleep_model: nn.Module, arousal_model: nn.Module, loader: torch.utils.data.DataLoader, 
                    device: torch.device, phase: int, epoch: int = 0, save_path: str = "."):
    """
    Unified validation and assessment loop for staging and arousal segmentation models.

    Args:
        sleep_model (nn.Module): Model for sleep staging.
        arousal_model (nn.Module): Model for arousal segmentation.
        loader (DataLoader): Validation dataset iterator.
        device (torch.device): CUDA or CPU execution device.
        phase (int): Training/Evaluation phase selector:
                     1: Staging only
                     2: Segmentation only
                     3 or 4: Combined multi-task evaluation
        epoch (int): Current training epoch index.
        save_path (str): Output directory path for visual reports.

    Returns:
        tuple: (results_dictionary, architecture_stats, metrics_summary)
    """
    if sleep_model: 
        sleep_model.eval()
    if arousal_model: 
        arousal_model.eval()
    
    results = {'s_true': [], 's_pred': [], 'a_true': [], 'a_pred': [], 's_probs': [], 'a_probs': []}
    a_stats = {'tp': 0, 'fp': 0, 'fn': 0}
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    ari_stats = {name: {'true_events': 0, 'pred_events': 0, 'epochs': 0} for name in stage_names.values()}

    with torch.no_grad():
        for x, y_s, y_a in loader:
            x, y_s, y_a = x.to(device), y_s.to(device), y_a.to(device)
            
            # --- 3-CHANNEL STAGING ROUTING ---
            b, s, c, t = x.shape  # Input Shape: (Batch, Sequence, Channels, Time_samples)
            x_flat = x.view(b * s, c, t)
            x_staging = x_flat[:, :3, :]  # Slice primary staging channels (EEG, EOG, EMG)
            
            logits_s = sleep_model(x_staging)  # Shape: (b * s, 5) or (b, s, 5)
            
            if phase in [1, 3, 4]:
                s_probs = torch.softmax(logits_s, dim=-1).cpu().numpy()
                s_pred = np.argmax(s_probs, axis=-1).flatten()
                
                results['s_true'].extend(y_s.cpu().numpy().flatten())
                results['s_pred'].extend(s_pred)
                results['s_probs'].extend(s_probs.reshape(-1, 5))
                
            # --- 4-CHANNEL AROUSAL SEGMENTATION ---
            if phase in [2, 3, 4] and arousal_model is not None:
                stage_context_tensor = torch.softmax(logits_s, dim=-1)
                logits_a = arousal_model(x, stage_context_tensor)
                
                a_prob = torch.sigmoid(logits_a).cpu().numpy()
                results['a_probs'].extend(a_prob)
                
                a_pred_raw = (a_prob > 0.5).astype(np.int8)
                y_a_np = y_a.cpu().numpy().astype(np.int8)
                y_s_np = y_s.cpu().numpy() 
                
                results['a_true'].extend(y_a_np.flatten())

                for b_idx in range(a_pred_raw.shape[0]):
                    # Post-process raw segmentation masks with clinical rules at 2Hz
                    p_mask_clinical = filter_duration(connect_events(a_pred_raw[b_idx], hz=2), hz=2)
                    t_mask = y_a_np[b_idx]
                    results['a_pred'].extend(p_mask_clinical.flatten())
                    
                    prec, rec, f1, tp, fp, fn = calculate_event_iou_metrics(t_mask, p_mask_clinical)
                    a_stats['tp'] += tp
                    a_stats['fp'] += fp
                    a_stats['fn'] += fn
                    
                    s_true_batch = y_s_np[b_idx]
                    for stg_idx in range(5):
                        ari_stats[stage_names[stg_idx]]['epochs'] += np.sum(s_true_batch == stg_idx)
                    
                    # Accumulate stage-specific predicted arousal events
                    labeled_pred, num_pred = label(p_mask_clinical)
                    for i in range(1, num_pred + 1):
                        idx = np.where(labeled_pred == i)[0][0] 
                        stg = s_true_batch[min(idx // 60, len(s_true_batch) - 1)]  # 60 samples per 30s epoch at 2Hz
                        ari_stats[stage_names[stg]]['pred_events'] += 1
                        
                    # Accumulate stage-specific ground truth arousal events
                    labeled_true, num_true = label(t_mask)
                    for i in range(1, num_true + 1):
                        idx = np.where(labeled_true == i)[0][0]
                        stg = s_true_batch[min(idx // 60, len(s_true_batch) - 1)]
                        ari_stats[stage_names[stg]]['true_events'] += 1

                # --- VISUALISATION SNAPSHOT (DOWNAMPLED TO 2Hz TIMELINE) ---
                snapshot_x = x[0, :, 0, ::50].reshape(-1).cpu().numpy()  # EEG channel subsampled by 50 (100Hz -> 2Hz)
                snapshot_t_mask = y_a_np[0].reshape(-1) 
                snapshot_p_mask = filter_duration(connect_events(a_pred_raw[0], hz=2), hz=2).reshape(-1) 
                snapshot_t_stages = y_s_np[0].reshape(-1)
                
                if phase == 4:
                    snapshot_p_stages = s_pred[:len(snapshot_t_stages)] 
                else:
                    snapshot_p_stages = np.argmax(torch.softmax(logits_s, dim=-1).cpu().numpy(), axis=-1)[0].reshape(-1)

    # --- METRICS GENERATION & CONSOLE SUMMARY ---
    metrics = {'macro_f1': 0.0, 'arousal_f1': 0.0, 'kappa': 0.0} 
    arch = None

    if phase in [1, 3, 4]:
        y_t_s = np.array(results['s_true'])
        s_probs_array = np.array(results['s_probs'])
        
        # 1. Optimise decision boundary using probability sweep
        optimal_n3_threshold = sweep_n3_boundary(y_t_s, s_probs_array)
        
        # 2. Generate final predictions using optimal boundary
        adjusted_preds = np.argmax(s_probs_array, axis=-1)
        upgrade_mask = (adjusted_preds == 2) & (s_probs_array[:, 3] >= optimal_n3_threshold)
        adjusted_preds[upgrade_mask] = 3
        
        results['s_pred'] = adjusted_preds.tolist()
        y_p_s = adjusted_preds
        
        stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
        s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(y_t_s, y_p_s, labels=[0, 1, 2, 3, 4], zero_division=0)
        
        metrics['macro_f1'] = float(np.mean(s_f1))
        metrics['kappa'] = float(cohen_kappa_score(y_t_s, y_p_s, labels=[0, 1, 2, 3, 4]))

        print("\n" + "[REPORT] PER-STAGE PERFORMANCE SUMMARY".center(55))
        print("-" * 55)
        print(f"{'Stage':<10} | {'Prec':<10} | {'Rec':<10} | {'F1':<10}")
        print("-" * 55)
        for i, name in enumerate(stages):
            print(f"{name:<10} | {s_prec[i]:<10.3f} | {s_rec[i]:<10.3f} | {s_f1[i]:<10.3f}")
        print("-" * 55)
        print(f"{'MACRO AVG':<10} | {np.mean(s_prec):<10.3f} | {np.mean(s_rec):<10.3f} | {metrics['macro_f1']:<10.3f}")
        print(f"{'COHEN KAPPA':<10} | {metrics['kappa']:<10.3f}") 
        print("-" * 55)

        arch = calculate_architecture_stats(y_t_s, y_p_s)

    if phase in [2, 3, 4]:
        y_a_true = np.array(results['a_true']).flatten()
        y_a_pred = np.array(results['a_pred']).flatten()
        
        sample_prec = precision_score(y_a_true, y_a_pred, zero_division=0)
        sample_rec = recall_score(y_a_true, y_a_pred, zero_division=0)
        metrics['arousal_f1'] = float(2 * a_stats['tp'] / (2 * a_stats['tp'] + a_stats['fp'] + a_stats['fn'] + 1e-6))

        clinical_stats = get_clinical_success_report(y_a_pred, y_a_true, fs=2)
        metrics['success_rate_80'] = clinical_stats['success_rate_80']
        metrics['onset_jitter'] = clinical_stats['mean_onset_jitter']

        print("\n" + "=" * 55)
        print("                 AROUSAL SEGMENTATION PERFORMANCE")
        print("=" * 55)
        print(f"{'Metric':<25} | {'Value':<15}")
        print("=" * 55)
        print(f"{'Sample Precision':<25} | {sample_prec:.4f}")
        print(f"{'Sample Recall':<25} | {sample_rec:.4f}")
        print(f"{'Event-Based F1 (IoU)':<25} | {metrics['arousal_f1']:.4f}")
        print(f"{'80% Overlap Success':<25} | {clinical_stats['success_rate_80'] * 100:.1f}%")
        print(f"{'Mean Onset Jitter':<25} | {clinical_stats['mean_onset_jitter']:.2f} sec")
        print(f"{'Median Onset Jitter':<25} | {clinical_stats['median_onset_jitter']:.2f} sec")
        print(f"{'Mean True Duration':<25} | {clinical_stats['mean_true_dur']:.1f} sec")
        print(f"{'Mean Pred Duration':<25} | {clinical_stats['mean_pred_dur']:.1f} sec")
        print("=" * 55)

    # --- PLOTTING ROUTINES ---
    try:
        if phase in [1, 3, 4]:
            plot_confusion_matrix(y_t_s, y_p_s, ['Wake', 'N1', 'N2', 'N3', 'REM'], epoch, save_path)
            plot_transition_matrix(y_t_s, y_p_s, ['Wake', 'N1', 'N2', 'N3', 'REM'], epoch, save_path)
            plot_hypnogram_comparison(results['s_true'], results['s_pred'], epoch, save_path)
            probs_array = np.array(results['s_probs'])
            plot_hypnodensity(probs_array, epoch, save_path)
            
        if phase in [2, 3, 4]:
            plot_arousal_confusion_matrix(y_a_true, y_a_pred, epoch, save_path)
            plot_event_raster(
                signals=snapshot_x, 
                true_mask=snapshot_t_mask, 
                pred_mask=snapshot_p_mask, 
                true_stages=snapshot_t_stages, 
                pred_stages=snapshot_p_stages, 
                epoch=epoch, 
                hz=2, 
                window_mins=5, 
                save_path=save_path
            )
            
    except Exception as e:
        print(f"[WARNING] Plotting sequence failed for Epoch {epoch + 1}. Error summary: {e}")

    return results, arch, metrics


# =====================================================================
# 3. CLINICAL VISUALISATION MODULES
# =====================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, epoch: int, save_path: str = "."):
    """Generates and saves a normalised confusion matrix heatmap for sleep stages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, vmin=0.0, vmax=1.0)
    plt.title(f'Sleep Staging Confusion Matrix - Epoch {epoch + 1}')
    plt.ylabel('Human Scorer')
    plt.xlabel('AI Prediction')
    plt.savefig(os.path.join(save_path, f"confusion_matrix_ep{epoch + 1}.png"), bbox_inches='tight')
    plt.close()


def plot_hypnogram_comparison(y_true: np.ndarray, y_pred: np.ndarray, epoch: int, save_path: str = "."):
    """Renders a comparative step plot of human vs AI sleep hypnograms over time."""
    mapping = {4: 0, 0: -1, 1: -2, 2: -3, 3: -4}  # Vertical plot order: REM, Wake, N1, N2, N3
    labels = ['REM', 'Wake', 'N1', 'N2', 'N3']
    ticks = [0, -1, -2, -3, -4]
    
    y_t, y_p = y_true[:500], y_pred[:500]  # Display first 500 epochs (~4 hours)
    true_mapped = [mapping[val] for val in y_t]
    pred_mapped = [mapping[val] for val in y_p]
    hours = np.arange(len(y_t)) * 30 / 3600  # Convert 30s epochs to hours
    
    plt.figure(figsize=(15, 5))
    plt.step(hours, true_mapped, color='black', linewidth=2, label='Human')
    plt.step(hours, pred_mapped, color='dodgerblue', alpha=0.7, linestyle='--', label='AI')
    plt.yticks(ticks, labels)
    plt.title(f'Hypnogram Comparison - Epoch {epoch + 1}')
    plt.xlabel('Time (Hours)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_path, f"hypnogram_ep{epoch + 1}.png"), bbox_inches='tight')
    plt.close()


def plot_hypnodensity(probs: np.ndarray, epoch: int, save_path: str = "."):
    """Plots a continuous probability area chart across all five sleep stages over time."""
    p = probs[:500]
    time_hours = np.arange(len(p)) * 30 / 3600
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    colors = ['#f4d03f', '#aed6f1', '#5dade2', '#2874a6', '#884ea0']
    
    plt.figure(figsize=(15, 6))
    plt.stackplot(time_hours, p.T, labels=labels, colors=colors, alpha=0.8)
    plt.title(f'Hypnodensity Probability Flow - Epoch {epoch + 1}')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xlim(0, time_hours[-1])
    plt.savefig(os.path.join(save_path, f"hypnodensity_ep{epoch + 1}.png"), bbox_inches='tight')
    plt.close()


def plot_bland_altman(true_vals: list, pred_vals: list, metric_name: str, unit: str, epoch: int, save_path: str = "."):
    """Generates a Bland-Altman agreement plot with mean bias and 95% confidence bounds."""
    true_vals, pred_vals = np.array(true_vals), np.array(pred_vals)
    if len(true_vals) < 2: 
        return 
        
    mean_val = (true_vals + pred_vals) / 2.0
    diff = pred_vals - true_vals
    md = np.mean(diff)
    sd = np.std(diff)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(mean_val, diff, alpha=0.5, color='dodgerblue', edgecolor='k')
    plt.axhline(md, color='black', linestyle='-', label=f'Mean Bias: {md:.2f}')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--', label=f'+1.96 SD: {md + 1.96 * sd:.2f}')
    plt.axhline(md - 1.96 * sd, color='red', linestyle='--', label=f'-1.96 SD: {md - 1.96 * sd:.2f}')
    plt.title(f'Bland-Altman Matrix: {metric_name} Agreement (Epoch {epoch + 1})')
    plt.xlabel(f'Mean {metric_name} ({unit})')
    plt.ylabel(f'Difference [AI - Human] ({unit})')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"bland_altman_{metric_name.lower().replace(' ', '_')}_ep{epoch + 1}.png"), bbox_inches='tight')
    plt.close()


def plot_transition_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, epoch: int, save_path: str = "."):
    """Compares state transition probability matrices between human scorer and AI model."""
    def get_matrix(data):
        matrix = np.zeros((5, 5))
        for i in range(len(data) - 1): 
            matrix[data[i], data[i + 1]] += 1
        return matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)

    tm_true, tm_pred = get_matrix(y_true), get_matrix(y_pred)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(tm_true, annot=True, fmt='.2f', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=ax[0], vmin=0.0, vmax=1.0)
    ax[0].set_title(f'Human Transitions (Epoch {epoch + 1})')
    ax[0].set_ylabel('From Stage')
    ax[0].set_xlabel('To Stage')
    
    sns.heatmap(tm_pred, annot=True, fmt='.2f', cmap='Purples', xticklabels=classes, yticklabels=classes, ax=ax[1], vmin=0.0, vmax=1.0)
    ax[1].set_title(f'AI Transitions (Epoch {epoch + 1})')
    ax[1].set_ylabel('From Stage')
    ax[1].set_xlabel('To Stage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"transition_matrix_ep{epoch + 1}.png"))
    plt.close()


def plot_event_raster(signals: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray, 
                      true_stages: np.ndarray, pred_stages: np.ndarray, epoch: int, 
                      hz: int = 2, window_mins: int = 5, save_path: str = "."):
    """Plots a comparative temporal raster of predicted vs ground-truth event locations."""
    samples = window_mins * 60 * hz
    t = np.arange(samples) / hz
    
    fig, ax = plt.subplots(figsize=(15, 4))
    sig_norm = (signals[:samples] - np.mean(signals[:samples])) / (np.std(signals[:samples]) + 1e-6)
    
    ax.plot(t, sig_norm, color='gray', alpha=0.3, label='Reference EEG Signal')
    ax.fill_between(t, 1.2, 1.8, where=true_mask[:samples] > 0, color='forestgreen', alpha=0.5, label='Human Scorer')
    ax.fill_between(t, 2.0, 2.6, where=pred_mask[:samples] > 0, color='dodgerblue', alpha=0.5, label='AI Prediction')
    
    ax.set_yticks([1.5, 2.3])
    ax.set_yticklabels(['Human', 'AI'])
    ax.set_ylim(-1.5, 3.0)
    ax.set_title(f'Event Location Agreement (Epoch {epoch + 1})')
    ax.set_xlabel('Time (Seconds)')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, f"raster_comparison_ep{epoch + 1}.png"), bbox_inches='tight')
    plt.close()


def plot_arousal_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, epoch: int, save_path: str):
    """Generates a binary confusion matrix heatmap for arousal event detection."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=['No Arousal', 'Arousal'],
                yticklabels=['No Arousal', 'Arousal'],
                vmin=0.0, vmax=1.0)
    
    plt.title(f'Arousal Confusion Matrix - Epoch {epoch + 1}')
    plt.ylabel('Human Scorer')
    plt.xlabel('AI Prediction')
    plt.savefig(os.path.join(save_path, f"arousal_confusion_matrix_ep{epoch + 1}.png"), bbox_inches='tight')
    plt.close()


def calculate_architecture_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates summary sleep architecture parameters (TST, SOL, WASO, SE) from epoch arrays.

    Args:
        y_true (np.ndarray): Ground truth stage labels.
        y_pred (np.ndarray): Model predicted stage labels.

    Returns:
        dict: Paired tuples (human, model) for TST, SOL, WASO, and SE metrics.
    """
    valid_mask = y_true != -100
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    
    total_epochs = len(yt)
    total_time_min = total_epochs * 0.5  # Each epoch represents 0.5 minutes (30 seconds)
    
    def get_sol(labels):
        sleep_indices = np.where(labels > 0)[0]  # First epoch of any sleep stage
        return sleep_indices[0] * 0.5 if len(sleep_indices) > 0 else total_time_min

    def get_waso(labels):
        sleep_indices = np.where(labels > 0)[0]
        if len(sleep_indices) < 2: 
            return 0.0
        onset, offset = sleep_indices[0], sleep_indices[-1]
        wake_after_onset = np.sum(labels[onset:offset] == 0)
        return wake_after_onset * 0.5

    tst_true = np.sum(yt > 0) * 0.5
    tst_pred = np.sum(yp > 0) * 0.5
    
    sol_true, sol_pred = get_sol(yt), get_sol(yp)
    waso_true, waso_pred = get_waso(yt), get_waso(yp)
    
    se_true = (tst_true / total_time_min) * 100.0
    se_pred = (tst_pred / total_time_min) * 100.0

    return {
        'TST': (tst_true, tst_pred),    # Total Sleep Time (minutes)
        'SOL': (sol_true, sol_pred),    # Sleep Onset Latency (minutes)
        'WASO': (waso_true, waso_pred), # Wake After Sleep Onset (minutes)
        'SE': (se_true, se_pred)        # Sleep Efficiency (%)
    }


def get_per_stage_arousal_stats(stage_mask: np.ndarray, arousal_mask: np.ndarray, hz: int = 2) -> dict:
    """
    Calculates the Arousal Index (events per hour of sleep) broken down by specific sleep stages.

    Args:
        stage_mask (np.ndarray): Sleep stage array.
        arousal_mask (np.ndarray): Binary arousal mask.
        hz (int): Mask sample frequency in Hz. Default is 2Hz.

    Returns:
        dict: Stage-wise counts and calculated Arousal Indices (ArI).
    """
    stages = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    stats = {name: {"count": 0, "total_minutes": 0.0} for name in stages.values()}
    
    unique, counts = np.unique(stage_mask, return_counts=True)
    for s_idx, count in zip(unique, counts):
        if s_idx in stages:
            stats[stages[s_idx]]["total_minutes"] = count * 0.5
            
    labeled_arousals, num_events = label(arousal_mask)
    for i in range(1, num_events + 1):
        event_indices = np.where(labeled_arousals == i)[0]
        start_stage = stage_mask[event_indices[0]]
        if start_stage in stages:
            stats[stages[start_stage]]["count"] += 1
            
    report = {}
    for name, data in stats.items():
        hours = data["total_minutes"] / 60.0
        ari = data["count"] / hours if hours > 0 else 0.0
        report[name] = {"ArI": round(ari, 2), "Count": data["count"]}
        
    return report


def perform_threshold_sweep(y_true_list: list, y_prob_list: list, hz: int = 2) -> float:
    """
    Sweeps segmentation decision thresholds (0.1 to 0.9) to locate the optimum event-level F1 score.

    Args:
        y_true_list (list): List of ground truth binary masks per batch.
        y_prob_list (list): List of predicted continuous probability masks per batch.
        hz (int): Array frequency in Hz. Default is 2Hz.

    Returns:
        float: Optimal threshold value yielding the maximum event-level IoU F1 score.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    sweep_results = []

    print("\n" + "[SWEEP] SEARCHING FOR CLINICAL OPTIMUM THRESHOLD".center(60))
    print("-" * 60)
    print(f"{'Thresh':<8} | {'Event F1':<10} | {'Prec':<8} | {'Rec':<8} | {'Pred Events':<12}")
    print("-" * 60)

    for thresh in thresholds:
        all_tp, all_fp, all_fn = 0, 0, 0
        total_pred_events = 0
        
        for b in range(len(y_prob_list)):
            p_mask = (y_prob_list[b] > thresh).astype(np.int8)
            p_mask_clinical = clinical_arousal_filter(p_mask, hz=hz)
            prec, rec, f1, tp, fp, fn = calculate_event_iou_metrics(y_true_list[b], p_mask_clinical)
            
            all_tp += tp
            all_fp += fp
            all_fn += fn
            
            _, count = label(p_mask_clinical)
            total_pred_events += count

        f1_event = 2 * all_tp / (2 * all_tp + all_fp + all_fn + 1e-6)
        p_event = all_tp / (all_tp + all_fp + 1e-6)
        r_event = all_tp / (all_tp + all_fn + 1e-6)
        
        print(f"{thresh:<8.1f} | {f1_event:<10.4f} | {p_event:<8.3f} | {r_event:<8.3f} | {total_pred_events:<12}")
        sweep_results.append((thresh, f1_event))

    best_thresh, _ = max(sweep_results, key=lambda x: x[1])
    print("-" * 60)
    print(f"[METRIC] OPTIMAL CLINICAL THRESHOLD DETERMINED: {best_thresh}")
    return best_thresh


def get_clinical_success_report(preds: np.ndarray, targets: np.ndarray, fs: int = 2) -> dict:
    """
    Evaluates clinical success metrics including 80% duration overlap success and temporal onset jitter.

    Args:
        preds (np.ndarray): Binary prediction array.
        targets (np.ndarray): Ground truth binary array.
        fs (int): Sample rate in Hz. Default is 2Hz.

    Returns:
        dict: Mean durations, 80% overlap success rate, and onset jitter statistics.
    """
    true_labels, n_true = label(targets > 0.5)
    pred_labels, n_pred = label(preds > 0.5)
    
    true_durations = [np.sum(true_labels == i) / fs for i in range(1, n_true + 1)]
    pred_durations = [np.sum(pred_labels == i) / fs for i in range(1, n_pred + 1)]
    
    success_80 = 0
    onset_jitters = []
    
    for i in range(1, n_true + 1):
        event_mask = (true_labels == i)
        true_onset = np.where(event_mask)[0][0]
        
        overlap_samples = (preds > 0.5) & event_mask
        overlap_ratio = np.sum(overlap_samples) / np.sum(event_mask)
        
        if overlap_ratio > 0:
            ai_event_id = pred_labels[np.where(overlap_samples)[0][0]]
            ai_onset = np.where(pred_labels == ai_event_id)[0][0]
            
            jitter = abs(true_onset - ai_onset) / fs  # Onset offset in seconds
            onset_jitters.append(jitter)
            
            if overlap_ratio >= 0.80:
                success_80 += 1
                
    return {
        'mean_true_dur': float(np.mean(true_durations)) if n_true > 0 else 0.0,
        'mean_pred_dur': float(np.mean(pred_durations)) if n_pred > 0 else 0.0,
        'success_rate_80': float(success_80 / n_true) if n_true > 0 else 0.0,
        'mean_onset_jitter': float(np.mean(onset_jitters)) if len(onset_jitters) > 0 else 0.0,
        'median_onset_jitter': float(np.median(onset_jitters)) if len(onset_jitters) > 0 else 0.0
    }
