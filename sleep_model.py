import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# Fallback check for optimised C++/CUDA Mamba kernels
try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


# =====================================================================
# LAYER INITIALISATION HELPERS
# =====================================================================

def create_fc(in_features: int, out_features: int, use_bias: bool = False) -> nn.Linear:
    """
    Creates a Fully Connected (Linear) layer with Kaiming Normal initialisation.
    Designed for linear or leaky activation pipelines.
    """
    layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
    if use_bias:
        nn.init.zeros_(layer.bias)
    return layer


def create_conv1d(in_channels: int, out_channels: int, kernel_size: int, 
                  stride: int = 1, padding: str = 'same') -> nn.Conv1d:
    """
    Creates a 1D Convolutional layer initialised for ReLU activations.
    Handles explicit padding calculation when stride > 1.
    """
    pad_val = padding if stride == 1 else (kernel_size // 2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad_val)
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    return layer


def create_max_pool1d(kernel_size: int, stride: int, padding: int = 0) -> nn.MaxPool1d:
    """Standard 1D Max Pooling builder."""
    return nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)


def create_batch_norm(num_features: int) -> nn.BatchNorm1d:
    """1D Batch Normalisation tuned for stability across signal features."""
    return nn.BatchNorm1d(num_features, eps=1e-3, momentum=0.01)


# =====================================================================
# FEATURE EXTRACTION MODULES
# =====================================================================

class SpectralFeatureLayer(nn.Module):
    """
    Extracts handcrafted domain-specific sleep features from raw raw signal arrays.
    Calculates power spectral density across standard EEG bands and extracts key amplitude ratios.
    
    Output Dimension: 11 distinct spectral scalar features per epoch.
    """
    def __init__(self, fs: int = 100):
        super().__init__()
        self.fs = fs
        # Frequency boundaries for standard sleep staging EEG bands (in Hz)
        self.bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'sigma': (12.0, 16.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 48.0)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (batch_size, channels=3, signal_length)
        Channel mapping assumed: 0 -> EEG, 1 -> EOG, 2 -> EMG
        """
        eeg = x[:, 0, :]  # Shape: (batch_size, signal_length)
        window_len = eeg.shape[-1]

        # 1. Compute Power Spectral Density (PSD) using Real FFT
        fft_values = torch.fft.rfft(eeg, n=window_len)  # Shape: (batch_size, window_len // 2 + 1)
        psd = torch.abs(fft_values) ** 2               # Shape: (batch_size, window_len // 2 + 1)
        
        # Frequency bins array
        freqs = torch.fft.rfftfreq(window_len, d=1/self.fs).to(x.device)

        # Calculate absolute band power for each defined range
        band_powers = []
        for name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_p = psd[:, mask].sum(dim=-1)            # Sum power across frequency mask
            band_powers.append(band_p)
            
        powers = torch.stack(band_powers, dim=-1)        # Shape: (batch_size, 6)
        total_power = psd.sum(dim=-1, keepdim=True) + 1e-9

        # 2. Extract Relative Powers and Clinical Signal Ratios
        rel_powers = powers / total_power               # 6 relative power features

        # Diagnostic ratios commonly used in sleep scoring
        theta_alpha_ratio = (powers[:, 1] / (powers[:, 2] + 1e-9)).unsqueeze(-1) # 1 feature
        delta_sigma_ratio = (powers[:, 0] / (powers[:, 3] + 1e-9)).unsqueeze(-1) # 1 feature
        
        # Root Mean Square (RMS) measurements across signal channels
        eeg_rms = torch.sqrt(torch.mean(x[:, 0, :]**2, dim=-1, keepdim=True) + 1e-9)
        eog_rms = torch.sqrt(torch.mean(x[:, 1, :]**2, dim=-1, keepdim=True) + 1e-9) # 1 feature
        emg_rms = torch.sqrt(torch.mean(x[:, 2, :]**2, dim=-1, keepdim=True) + 1e-9) # 1 feature
        
        eog_eeg_ratio = eog_rms / (eeg_rms + 1e-9)                                  # 1 feature
        
        # 3. Concatenate all features into a single 11-dimensional expert vector
        expert_vector = torch.cat([
            rel_powers,         # 6 features
            theta_alpha_ratio,  # 1 feature
            delta_sigma_ratio,  # 1 feature
            eog_rms,            # 1 feature
            emg_rms,            # 1 feature
            eog_eeg_ratio       # 1 feature
        ], dim=-1)              # Final Shape: (batch_size, 11)
        
        return expert_vector


class TemporalAttention(nn.Module):
    """
    Applies temporal self-attention across 1D feature maps to pool 
    time steps into a single representative feature vector.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(0.2), 
            nn.Conv1d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, channels, sequence_length)
        attn_logits = self.attention(x)               # Shape: (batch_size, 1, sequence_length)
        weights = F.softmax(attn_logits, dim=-1)      # Normalise weights over time dimension
        
        # Weighted sum reduction over the time axis
        pooled = (x * weights).sum(dim=-1)            # Output Shape: (batch_size, channels)
        return pooled


class MultiScaleTower(nn.Module):
    """
    Extracts deep morphological features from a single signal channel 
    using varying convolutional filter sizes (broad vs fine granularities).
    """
    def __init__(self, fs: int = 100):
        super().__init__()
        # Block 1: Broad macro-morphology patterns (large receptive field)
        self.conv1 = create_conv1d(1, fs, kernel_size=21, stride=5) 
        self.bn1 = create_batch_norm(fs)
        self.conv2 = create_conv1d(fs, fs, kernel_size=21, stride=1)
        self.bn2 = create_batch_norm(fs)
        self.pool1 = create_max_pool1d(kernel_size=2, stride=2)

        # Block 2: Detailed mid-level patterns
        self.conv3 = create_conv1d(fs, fs * 2, kernel_size=5, stride=1)
        self.bn3 = create_batch_norm(fs * 2)
        self.conv4 = create_conv1d(fs * 2, fs * 2, kernel_size=5, stride=1)
        self.bn4 = create_batch_norm(fs * 2)
        self.pool2 = create_max_pool1d(kernel_size=2, stride=2)

        # Block 3: Fine high-dimensional nuance
        self.conv5 = create_conv1d(fs * 2, fs * 4, kernel_size=5, stride=1)
        self.bn5 = create_batch_norm(fs * 4)
        self.conv6 = create_conv1d(fs * 4, fs * 4, kernel_size=5, stride=1)
        self.bn6 = create_batch_norm(fs * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 1, raw_signal_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        # Output shape: (batch_size, fs * 4, reduced_time_steps)
        return x


# =====================================================================
# MAMBA STATE-SPACE MODEL SEQUENCING
# =====================================================================

class SimpleMambaBlock(nn.Module):
    """
    Single directional Mamba State-Space Model block.
    Uses native C++ mamba_ssm if installed; falls back to PyTorch implementation.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.use_fast_mamba = HAS_MAMBA_SSM
        self.norm = nn.LayerNorm(d_model)
        
        if self.use_fast_mamba:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            # Fallback pure PyTorch State-Space implementation
            self.d_inner = int(expand * d_model)
            self.d_state = d_state
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
            self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
            self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, d_model)
        z = self.norm(x)
        
        if self.use_fast_mamba:
            out = self.mamba(z)
        else:
            # Step 1: Input projection split into path processing and residual gate
            z_res = self.in_proj(z)
            z_val, res = z_res.split(self.d_inner, dim=-1)
            
            # Step 2: 1D Depthwise Convolution
            z_val = z_val.transpose(1, 2)
            z_val = F.silu(self.conv1d(z_val)[:, :, :z.shape[1]])
            z_val = z_val.transpose(1, 2)
            
            # Step 3: Parameter discretisations (delta, B, C)
            z_dbl = self.x_proj(z_val)
            delta, B, C = z_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(delta))
            A = -torch.exp(self.A_log.float())
            
            # Step 4: Scan recurrence loop across sequence length
            y_list = []
            h = torch.zeros(z.shape[0], self.d_inner, self.d_state, device=z.device)
            
            for t in range(z.shape[1]):
                dt = delta[:, t, :].unsqueeze(-1)
                dA = torch.exp(dt * A)
                dB = dt * B[:, t, :].unsqueeze(1)
                h = dA * h + dB * z_val[:, t, :].unsqueeze(-1)
                y_step = torch.matmul(h, C[:, t, :].unsqueeze(-1)).squeeze(-1)
                y_list.append(y_step)
            
            # Step 5: Output gating and projection
            y = torch.stack(y_list, dim=1) + z_val * self.D
            out = self.out_proj(y * F.silu(res))
            
        return x + out  # Residual connection


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba wrapper processing sequences in forward and reverse passes."""
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.mamba_fwd = SimpleMambaBlock(d_model, d_state)
        self.mamba_bwd = SimpleMambaBlock(d_model, d_state)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, d_model)
        out_fwd = self.mamba_fwd(x)
        
        # Reverse along sequence dimension
        x_bwd = torch.flip(x, dims=[1]) 
        out_bwd = torch.flip(self.mamba_bwd(x_bwd), dims=[1])
        
        # Merge directions
        return self.out_proj(torch.cat([out_fwd, out_bwd], dim=-1))


class StackedBiMamba(nn.Module):
    """Stack of multiple Bidirectional Mamba blocks with normalisation and dropout."""
    def __init__(self, d_model: int, n_layers: int = 3, d_state: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([BiMambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, d_model)
        for i, layer in enumerate(self.layers):
            z = self.norms[i](x)
            z = layer(z)
            x = x + self.dropout(z)  # Layer residual connection
        return x


# =====================================================================
# FULL TOP-LEVEL MODEL ARCHITECTURE
# =====================================================================

class SleepStagingBaseline3Ch(nn.Module):
    """
    3-Channel Sleep Staging Architecture (EEG, EOG, EMG).
    Combines deep spatial CNN features, handcrafted FFT spectral vectors, 
    and clinical scalar metrics into a Bidirectional Mamba sequence model.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 5, d_model: int = 256, n_layers: int = 3):
        super().__init__()
        fs = 100  # Target sampling rate in Hz
        
        # Channel feature towers
        self.tower_eeg = MultiScaleTower(fs=fs)
        self.tower_eog = MultiScaleTower(fs=fs)
        self.tower_emg = MultiScaleTower(fs=fs)
        
        # Temporal attention aggregators per signal channel
        self.attn_eeg = TemporalAttention(fs * 4)
        self.attn_eog = TemporalAttention(fs * 4)
        self.attn_emg = TemporalAttention(fs * 4)

        # FFT Spectral domain extraction
        self.spectral_layer = SpectralFeatureLayer(fs=fs)
        self.spec_norm = nn.BatchNorm1d(11)  # Matches 11 spectral features output
        
        # Combined Feature Dimension Calculation:
        # - CNN spatial features: 3 channels * (fs * 4) = 1200
        # - Selected spectral features: 11
        # - Metadata/Clinical scalars: 3 (Age, Sex, BMI)
        # Total dimension = 1214
        in_dim = (fs * 4 * 3) + 11 + 3

        self.projection = nn.Sequential(
            nn.Linear(in_dim, 512), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Long-range sequence modeling over sequential epochs
        self.sequence_model = StackedBiMamba(d_model=d_model, n_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, clinical_scalars: torch.Tensor = None) -> torch.Tensor:
        """
        Accepts 3D single-batch or 4D multi-epoch sequence inputs.
        
        Shapes:
            x (4D): (Batch, Sequence, Channels=3, Signal_Length)
            x (3D): (Total_Epochs, Channels=3, Signal_Length)
            clinical_scalars: (Batch, Sequence, 3) or (Total_Epochs, 3)
        """
        # Step 1: Input Shape Normalisation & Batch/Sequence Flattening
        if len(x.shape) == 4:
            B, S, C, T = x.shape
            x_flat = x.view(B * S, C, T)
            if clinical_scalars is None:
                scalars_flat = torch.zeros((B * S, 3), device=x.device, dtype=x.dtype)
            else:
                scalars_flat = clinical_scalars.view(B * S, 3)
        else:
            total_epochs, C, T = x.shape
            x_flat = x
            B, S = 1, total_epochs
            if clinical_scalars is None:
                scalars_flat = torch.zeros((B * S, 3), device=x.device, dtype=x.dtype)
            else:
                scalars_flat = clinical_scalars.view(B * S, 3)

        # Step 2: Extract CNN Spatial Features per Channel (Path A)
        # Flattened inputs shapes: (B*S, 1, Signal_Length)
        eeg_feat = self.attn_eeg(self.tower_eeg(x_flat[:, 0:1, :])) # Shape: (B*S, 400)
        eog_feat = self.attn_eog(self.tower_eog(x_flat[:, 1:2, :])) # Shape: (B*S, 400)
        emg_feat = self.attn_emg(self.tower_emg(x_flat[:, 2:3, :])) # Shape: (B*S, 400)
        
        # Step 3: Compute FFT Spectral Expert Features (Path B)
        spec_feat = self.spectral_layer(x_flat)                       # Shape: (B*S, 11)
        spec_feat = torch.log1p(spec_feat)                            # Compress dynamic range
        spec_feat = self.spec_norm(spec_feat)                         # Normalise features
        
        # Step 4: Concatenate Feature Pathways & Project to Hidden Space (Path C)
        combined = torch.cat([eeg_feat, eog_feat, emg_feat, spec_feat, scalars_flat], dim=-1) # Shape: (B*S, 1214)
        
        # Reshape projection output back into sequence representation
        projected = self.projection(combined).view(B, S, -1)           # Shape: (B, S, d_model)
        
        # Step 5: Sequence Modeling & Final Stage Classification
        seq_out = self.sequence_model(projected)                      # Shape: (B, S, d_model)
        logits = self.classifier(seq_out)                             # Shape: (B, S, n_classes)
        
        return logits
