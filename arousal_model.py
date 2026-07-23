import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# =====================================================================
# LAYER INITIALISATION HELPERS
# =====================================================================

def create_conv1d(in_channels: int, out_channels: int, kernel_size: int, 
                  stride: int = 1, padding: str = 'same', dilation: int = 1) -> nn.Conv1d:
    """
    Creates a 1D Convolutional layer with Kaiming Normal initialisation tailored for ReLU activations.
    Computes explicit padding bounds when stride > 1.
    """
    pad_val = padding if stride == 1 else (kernel_size // 2)
    layer = nn.Conv1d(
        in_channels, out_channels, kernel_size, 
        stride=stride, padding=pad_val, dilation=dilation
    )
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    return layer


# =====================================================================
# SPECTRAL FEATURE EXTRACTION ENGINE
# =====================================================================

class SpectralExpertLayer(nn.Module):
    """
    Extracts time-frequency spectral features and Hjorth mobility metrics 
    from electrophysiological channels (EEG, EOG, EMG).

    Calculates STFT power maps across standard EEG bands and derives key 
    spectral ratios and EMG power bands for arousal segmentation.
    
    Output Dimension: 10 expert feature maps per time step.
    """
    def __init__(self, fs: int = 100, n_fft: int = 200, hop_length: int = 100):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Frequency band boundaries mapped to STFT bin indices (100 Hz sampling rate, 200 n_fft)
        self.bands = {
            'delta': (1, 8),    # 0.5 - 4.0 Hz
            'theta': (8, 16),   # 4.0 - 8.0 Hz
            'alpha': (16, 26),  # 8.0 - 13.0 Hz
            'sigma': (24, 32),  # 12.0 - 16.0 Hz
            'beta':  (26, 60),  # 13.0 - 30.0 Hz
            'gamma': (60, 96)   # 30.0 - 48.0 Hz
        }
        self.out_dim = 10 
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (Batch * Sequence, Channels=4, Signal_Samples)
        Expected Channel Indexing: 0=EEG, 1=EOG, 2=EMG, 3=ECG
        """
        eeg = x[:, 0, :]  # Shape: (B*S, Signal_Samples)
        eog = x[:, 1, :]  # Shape: (B*S, Signal_Samples)
        emg = x[:, 2, :]  # Shape: (B*S, Signal_Samples)

        def get_mobility(sig: torch.Tensor) -> torch.Tensor:
            """Calculates Hjorth Mobility: square root of variance of first derivative over variance of signal."""
            dsig = sig[:, 1:] - sig[:, :-1]
            return torch.sqrt(torch.var(dsig, dim=-1) / (torch.var(sig, dim=-1) + 1e-6))

        # Calculate signal mobility across EEG and EOG channels
        mob_eeg = get_mobility(eeg)  # Shape: (B*S,)
        mob_eog = get_mobility(eog)  # Shape: (B*S,)

        # Compute Short-Time Fourier Transform (STFT) on EEG
        window = self.window
        stft_eeg = torch.stft(
            eeg, n_fft=self.n_fft, hop_length=self.hop_length, 
            win_length=self.n_fft, window=window, return_complex=True, center=False
        )  # Shape: (B*S, Freq_Bins, Time_Steps)
        p_eeg = torch.abs(stft_eeg)**2
        
        # Calculate power per frequency band
        b = {n: p_eeg[:, lim[0]:lim[1], :].sum(dim=1) for n, lim in self.bands.items()}
        
        ratio_ab = b['alpha'] / (b['beta'] + 1e-6)
        ratio_as = b['alpha'] / (b['sigma'] + 1e-6)
        ratio_ad = b['alpha'] / (b['delta'] + 1e-6)
        ratio_bg = b['beta'] / (b['gamma'] + 1e-6)
        abs_alpha = b['alpha']

        # Compute STFT on EMG channel
        stft_emg = torch.stft(
            emg, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, return_complex=True, center=False
        )
        p_emg = torch.abs(stft_emg)**2
        
        emg_delta_rel = p_emg[:, 1:8, :].sum(dim=1) / (p_emg.sum(dim=1) + 1e-6)
        emg_delta_abs = p_emg[:, 1:8, :].sum(dim=1)
        emg_gamma_abs = p_emg[:, 60:96, :].sum(dim=1)

        time_dim = p_eeg.shape[-1]
        mob_eeg = mob_eeg.unsqueeze(-1).expand(-1, time_dim)  # Expand to match STFT time dimensions
        mob_eog = mob_eog.unsqueeze(-1).expand(-1, time_dim)

        # Concatenate into 10-dimensional feature map matrix
        experts = torch.stack([
            mob_eeg, mob_eog, ratio_ab, ratio_as, ratio_ad, 
            ratio_bg, abs_alpha, emg_delta_rel, emg_delta_abs, emg_gamma_abs
        ], dim=1)  # Output Shape: (B*S, 10, Time_Steps)

        return experts


# =====================================================================
# FULLY CONVOLUTIONAL 1D TIME-SERIES U-NET
# =====================================================================

class ArousalSegmentationModel4Ch(nn.Module):
    """
    1D U-Net Architecture for micro-arousal event segmentation.
    Fuses raw 4-channel electrophysiological signals with handcrafted 
    spectral features and sleep-stage context probabilities.
    """
    def __init__(self, n_channels: int = 4, n_classes: int = 5, fs: int = 100): 
        super().__init__()
        self.fs = fs
        
        # 1. Convolutional Encoder Path
        self.enc_conv1 = create_conv1d(n_channels, 16, kernel_size=9)
        self.enc_pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.enc_conv2 = create_conv1d(16, 32, kernel_size=9)
        self.enc_pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.enc_conv3 = create_conv1d(32, 64, kernel_size=9)
        self.enc_pool3 = nn.MaxPool1d(kernel_size=2)
        
        # 2. Context Fusion Processing
        self.expert_layer = SpectralExpertLayer(fs=fs)
        self.expert_proj = create_conv1d(self.expert_layer.out_dim, 32, kernel_size=1)
        self.stage_context_proj = nn.Linear(n_classes, 32)
        self.context_dropout = nn.Dropout(p=0.5)
        
        # 3. Bottleneck Core Layer (64 CNN + 32 Stage Context + 32 Spectral Experts = 128 channels)
        self.bottleneck = create_conv1d(128, 128, kernel_size=9)
        
        # 4. Convolutional Decoder Path (with skip connections)
        self.dec_conv1 = create_conv1d(128 + 64, 64, kernel_size=9)  # 128 upsampled + 64 skip
        self.dec_conv2 = create_conv1d(64 + 32, 32, kernel_size=9)   # 64 upsampled + 32 skip
        self.dec_conv3 = create_conv1d(32 + 16, 16, kernel_size=9)   # 32 upsampled + 16 skip
        
        # 5. Output Prediction Head (Pulls 100 Hz signal down to 2 Hz decision points)
        self.caiser_2hz_pool = nn.AvgPool1d(kernel_size=50)  # 100 Hz / 50 = 2 Hz resolution
        self.final_head = nn.Conv1d(16, 1, kernel_size=1) 
        
        # Initialise bias to -2.94 to reflect low class prevalence (~5% positive samples)
        nn.init.constant_(self.final_head.bias, -2.94)

    def forward(self, x: torch.Tensor, stage_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input signals. Shape: (Batch, Sequence, Channels=4, Samples=3000)
            stage_probs (torch.Tensor): Stage context probabilities. Shape: (Batch, Sequence, Classes=5)

        Returns:
            torch.Tensor: Logit predictions mapped to 2 Hz temporal resolution. Shape: (Batch, Sequence, 60)
        """
        b, s, c, t = x.shape
        x_reshaped = x.view(b * s, c, t)  # Shape: (B*S, 4, 3000)
        
        # --- ENCODER PATH ---
        x1 = F.elu(self.enc_conv1(x_reshaped))  # Shape: (B*S, 16, 3000)
        p1 = self.enc_pool1(x1)                 # Shape: (B*S, 16, 1500)
        
        x2 = F.elu(self.enc_conv2(p1))          # Shape: (B*S, 32, 1500)
        p2 = self.enc_pool2(x2)                 # Shape: (B*S, 32, 750)
        
        x3 = F.elu(self.enc_conv3(p2))          # Shape: (B*S, 64, 750)
        p3 = self.enc_pool3(x3)                 # Shape: (B*S, 64, 375)
        
        # --- BOTTLENECK FEATURE FUSION ---
        target_seq_len = p3.shape[-1]  # 375 time steps
        
        # Process handcrafted spectral features
        raw_experts = self.expert_layer(x_reshaped)  # Shape: (B*S, 10, Time_Steps)
        experts_resampled = F.interpolate(raw_experts, size=target_seq_len, mode='linear', align_corners=False)
        expert_ctx = F.elu(self.expert_proj(experts_resampled))  # Shape: (B*S, 32, 375)
        
        # Process stage probabilities context
        stage_ctx = F.elu(self.stage_context_proj(stage_probs.view(b * s, 5)))
        stage_ctx = stage_ctx.unsqueeze(-1).expand(-1, -1, target_seq_len)  # Shape: (B*S, 32, 375)
        stage_ctx = self.context_dropout(stage_ctx)
        
        # Merge bottleneck channels
        merged_bottleneck = torch.cat([p3, stage_ctx, expert_ctx], dim=1)  # Shape: (B*S, 128, 375)
        b_core = F.elu(self.bottleneck(merged_bottleneck))                 # Shape: (B*S, 128, 375)
        
        # --- DECODER PATH WITH SKIP CONNECTIONS ---
        u1 = F.interpolate(b_core, size=750, mode='nearest')
        u1 = u1[:, :, :x3.shape[-1]] 
        merge1 = torch.cat([u1, x3], dim=1)                                # Shape: (B*S, 192, 750)
        d1 = F.elu(self.dec_conv1(merge1))                                 # Shape: (B*S, 64, 750)
        
        u2 = F.interpolate(d1, size=1500, mode='nearest')
        u2 = u2[:, :, :x2.shape[-1]]
        merge2 = torch.cat([u2, x2], dim=1)                                # Shape: (B*S, 96, 1500)
        d2 = F.elu(self.dec_conv2(merge2))                                 # Shape: (B*S, 32, 1500)
        
        u3 = F.interpolate(d2, size=3000, mode='nearest')
        u3 = u3[:, :, :x1.shape[-1]]
        merge3 = torch.cat([u3, x1], dim=1)                                # Shape: (B*S, 48, 3000)
        d3 = F.elu(self.dec_conv3(merge3))                                 # Shape: (B*S, 16, 3000)
        
        # --- OUTPUT MAPPING ---
        pooled_features = self.caiser_2hz_pool(d3)                        # Shape: (B*S, 16, 60)
        logits = self.final_head(pooled_features)                          # Shape: (B*S, 1, 60)
        
        return logits.view(b, s, 60)
    
    @torch.no_grad()
    def predict_events(self, x: torch.Tensor, stage_probs: torch.Tensor, 
                       onset_thresh: float = 0.70, offset_thresh: float = 0.35, 
                       max_duration_sec: float = 45.0) -> list:
        """
        Parses continuous probabilities into clinical event intervals using hysteresis thresholds 
        (Schmidt trigger logic) and enforces duration bounds.

        Returns:
            list: List of dictionaries containing event parameters: [{'Name', 'Start', 'Duration'}]
        """
        logits = self.forward(x, stage_probs)  # Output Shape: (B, S, 60)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        fs_output = 2  # Output resolution: 2 Hz (2 frames per second)
        events = []
        in_arousal = False
        start_frame = 0
        
        # Asymmetric threshold tracking loop
        for i, p in enumerate(probs):
            if not in_arousal:
                # Trigger event onset
                if p >= onset_thresh:
                    in_arousal = True
                    start_frame = i
            else:
                # Trigger event offset
                if p < offset_thresh:
                    in_arousal = False
                    duration_frames = i - start_frame
                    duration_sec = duration_frames / fs_output
                    
                    # Validate event duration limits (3s to 45s)
                    if 3.0 <= duration_sec <= max_duration_sec:
                        events.append({
                            'Name': 'Arousal',
                            'Start': start_frame / fs_output,
                            'Duration': duration_sec
                        })
                        
        # Handle trailing end-of-record events
        if in_arousal:
            duration_sec = (len(probs) - start_frame) / fs_output
            if 3.0 <= duration_sec <= max_duration_sec:
                events.append({
                    'Name': 'Arousal',
                    'Start': start_frame / fs_output,
                    'Duration': duration_sec
                })
                
        return events


# =====================================================================
# DURATION-PENALISED FOCAL TVERSKY LOSS
# =====================================================================

class DurationPenalizedFocalTverskyLoss(nn.Module):
    """
    Combines Focal Tversky Loss with an explicit continuous-plateau penalty.
    Penalises extended false-positive prediction blocks to enforce clear event boundaries.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0, 
                 smooth: float = 1e-6, roi_expansion: int = 0, 
                 max_duration_sec: float = 15.0, fs_output: int = 2, penalty_weight: float = 5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.roi_expansion = roi_expansion
        
        self.penalty_weight = penalty_weight
        self.max_frames = int(max_duration_sec * fs_output)  # 15s * 2 Hz = 30 frames

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs. Shape: (Batch, Sequence, 60)
            targets (torch.Tensor): Ground truth masks. Shape: (Batch, Sequence, 60)
        """
        probs = torch.sigmoid(logits)
        B, S, T = targets.shape
        
        # --- PHASE 1: SPATIAL FOCAL TVERSKY LOSS ---
        targets_spatial = targets.view(B * S, 1, T)
        
        roi_mask = F.max_pool1d(
            F.pad(targets_spatial, (self.roi_expansion, self.roi_expansion), value=0),
            kernel_size=(self.roi_expansion * 2) + 1,
            stride=1, 
            padding=0
        )
        
        roi_mask = roi_mask.view(-1)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        spatial_weights = roi_mask + 0.1
        
        TP = (probs_flat * targets_flat * spatial_weights).sum()
        FP = ((1.0 - targets_flat) * probs_flat * spatial_weights).sum()
        FN = (targets_flat * (1.0 - probs_flat) * spatial_weights).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1.0 - tversky_index) ** self.gamma
        
        # --- PHASE 2: DURATION PLATEAU PENALTY ---
        probs_spatial = probs.view(B * S, 1, T)
        
        # Pad bounds to preserve timeline length
        pad_left = self.max_frames // 2
        pad_right = self.max_frames - pad_left - 1
        padded_probs = F.pad(probs_spatial, (pad_left, pad_right), mode='constant', value=0.0)
        
        # Local averaging density map
        plateau_density = F.avg_pool1d(padded_probs, kernel_size=self.max_frames, stride=1)
        
        # Isolate and penalise false-positive prediction plateaus
        false_positive_plateaus = plateau_density * (1.0 - targets_spatial)
        duration_penalty = false_positive_plateaus.mean()
        
        return focal_tversky + (self.penalty_weight * duration_penalty)


# =====================================================================
# VERIFICATION & TEST HARNESS
# =====================================================================

if __name__ == "__main__":
    batch_size = 2
    epochs_per_file = 4
    channels = 4
    samples_per_epoch = 3000
    
    # Instantiate model with 4-channel layout
    model = ArousalSegmentationModel4Ch(n_channels=4, n_classes=5)
    
    mock_psg_batch = torch.randn(batch_size, epochs_per_file, channels, samples_per_epoch)
    mock_stage_probs = torch.randn(batch_size, epochs_per_file, 5)
    
    output_logits = model(mock_psg_batch, mock_stage_probs)
    print("Execution Graph Shape Pass Verification:")
    print(" -> Input Tensors Shape: ", mock_psg_batch.shape)
    print(" -> Output Logits Shape (Expected [2, 4, 60]):", output_logits.shape)
    
    # Test asymmetric threshold prediction engine
    parsed_events = model.predict_events(mock_psg_batch, mock_stage_probs)
    print(f" -> Successfully parsed {len(parsed_events)} clinical event windows.")
