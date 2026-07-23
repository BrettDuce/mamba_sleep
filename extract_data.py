import mne
import torch
import numpy as np
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================

FS = 100                  # Target downsampling frequency (Hz)
EPOCH_SEC = 30           # Duration of a single sleep epoch (seconds)
SAMPLES = FS * EPOCH_SEC  # Target samples per 30-second epoch (100 Hz * 30 s = 3000 samples)
EDF_PATH = r"F:\Data\Untrimmed"     # Directory containing raw .EDF files
LABEL_PATH = r"F:\Data\Untrimmed"   # Directory containing Profusion XML annotation files
SAVE_PATH = r"F:\Data\TinyFeatures" # Output directory for processed PyTorch tensors

# CHANNEL DERIVATION ALIASES & BIPOLAR PAIRINGS
DERIVATION_MAP = {
    'C4-M1': {
        'aliases': ['EEG C4-M1', 'C4-M1', 'C4-A1', 'EEG C4-A1'],
        'anode': ['EEG C4', 'C4'],
        'cathode': ['EEG M1', 'M1', 'A1']
    },
    'E2-M1': {
        'aliases': ['EOG E2-M1', 'E2-M1', 'E2-M2', 'ROC-A1', 'E2-A1'],
        'anode': ['EOG E2', 'ROC', 'E2'],
        'cathode': ['EEG M1', 'M1', 'A1']
    },
    'Chin1-ChinZ': {
        'aliases': ['EMG Chin', 'Chin1-ChinZ', 'Chin 1-Chin Z', 'Chin1-Chin2', 'EMG'],
        'anode': ['Chin1', 'Chin 1'],
        'cathode': ['Chin2', 'Chin 2', 'Chin Z', 'ChinZ']
    }
}

os.makedirs(SAVE_PATH, exist_ok=True)


# =====================================================================
# MONTAGE STANDARDISED DERIVATION ENGINE
# =====================================================================

def standardize_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Standardises input EDF channel names against the derivation map.
    Renames pre-derived channels or dynamically computes bipolar references 
    from single-ended electrodes (e.g., C4 and M1 -> C4-M1).

    Args:
        raw (mne.io.BaseRaw): Loaded MNE raw EDF object.

    Returns:
        mne.io.BaseRaw: Raw object containing standardized channel labels.
    """
    for target_name, config in DERIVATION_MAP.items():
        # Search for direct pre-referenced channel aliases
        existing_alias = next((ch for ch in raw.ch_names if ch in config['aliases']), None)
        
        if existing_alias:
            mne.rename_channels(raw.info, {existing_alias: target_name})
            continue 
        
        # Search for unreferenced anode and cathode pairs
        anode = next((ch for ch in raw.ch_names if ch in config['anode']), None)
        cathode = next((ch for ch in raw.ch_names if ch in config['cathode']), None)
        
        if anode and cathode:
            try:
                raw = mne.set_bipolar_reference(
                    raw, anode, cathode, ch_name=target_name, 
                    drop_refs=False, verbose=False
                )
            except Exception as e:
                print(f"      Could not derive {target_name}: {e}")
                
    return raw


# =====================================================================
# PROFUSION XML PARSING & ANNOTATION EXTRACTION
# =====================================================================

def map_profusion_codes(stages_list: list) -> np.ndarray:
    """
    Maps Compumedics Profusion XML numerical stage codes to standard indices:
    Profusion mapping: 0=Wake, 1=N1, 2=N2, 3=N3, 4=N4 (merged into N3), 5=REM.

    Args:
        stages_list (list): Raw XML stage integers.

    Returns:
        np.ndarray: Standardised stage array where 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
    """
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4} 
    return np.array([mapping.get(s, 0) for s in stages_list])


def parse_profusion_xml(xml_path: str) -> tuple:
    """
    Parses Compumedics Profusion XML files to extract epoch sleep stage arrays 
    and event metadata dataframes.

    Args:
        xml_path (str): Path to the target XML annotation file.

    Returns:
        tuple: (cleaned_stages_array, events_dataframe)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        epoch_len_tag = root.find('.//EpochLength')
        epoch_len = 30
        if epoch_len_tag is not None:
            epoch_len = int(epoch_len_tag.text)
            if epoch_len != 30:
                print(f"(!) INFO: XML specifies {epoch_len}s epochs. Adjusting...")

        raw_stages = [int(s.text) for s in root.findall('.//SleepStage')]
        cleaned_stages = map_profusion_codes(raw_stages)
        
        events = []
        for event in root.findall('.//ScoredEvent'):
            if event.find('Start') is None: 
                continue
            events.append({
                'name': event.find('Name').text if event.find('Name') is not None else "Unknown",
                'start': float(event.find('Start').text),
                'duration': float(event.find('Duration').text),
                'input': event.find('Input').text if event.find('Input') is not None else "N/A"
            })
            
        return cleaned_stages, pd.DataFrame(events)
        
    except Exception as e:
        print(f"❌ XML Error {os.path.basename(xml_path)}: {e}")
        return np.array([]), pd.DataFrame()


def create_arousal_mask(events_df: pd.DataFrame, total_epochs: int, 
                        epoch_sec: int = 30, target_hz: int = 10) -> np.ndarray:
    """
    Constructs a 2D sample-level binary arousal mask from event onset timestamp records.

    Args:
        events_df (pd.DataFrame): Dataframe containing scored event intervals.
        total_epochs (int): Total number of synchronized study epochs.
        epoch_sec (int): Epoch duration in seconds. Default is 30s.
        target_hz (int): Target mask sampling frequency. Default is 10Hz.

    Returns:
        np.ndarray: Reshaped binary arousal array. Shape: (total_epochs, target_hz * epoch_sec)
    """
    samples_per_epoch = epoch_sec * target_hz  # e.g., 30s * 10Hz = 300 samples per epoch
    mask = np.zeros(total_epochs * samples_per_epoch, dtype=np.float32)
    
    if not events_df.empty:
        # Filter dataframe for arousal event markers
        arousals = events_df[events_df['name'].str.contains('arousal', case=False, na=False)]
        for _, row in arousals.iterrows():
            start_sample = int(row['start'] * target_hz)
            end_sample = int((row['start'] + row['duration']) * target_hz)
            
            # Bound array pointers within synchronized study limits
            end_sample = min(end_sample, len(mask))
            if start_sample < len(mask):
                mask[start_sample:end_sample] = 1.0
        
    return mask.reshape(total_epochs, samples_per_epoch)


# =====================================================================
# STUDY PREPROCESSING & FEATURE EXTRACTION PIPELINE
# =====================================================================

def preprocess_study(edf_path: str, xml_path: str) -> dict:
    """
    Executes raw signal loading, montage standardisation, digital filtering, 
    downsampling, normalization, and epoch segmentation.

    Args:
        edf_path (str): File path to raw input .EDF recording.
        xml_path (str): File path to matching Profusion XML annotations.

    Returns:
        dict: Preprocessed dictionary containing PyTorch tensors:
              - 'x': Signal tensor. Shape: (num_epochs, 3_channels, 3000_samples)
              - 'y_stage': Sleep stage label tensor. Shape: (num_epochs,)
              - 'y_arousal': Arousal mask tensor. Shape: (num_epochs, 300_samples)
    """
    # 1. Parse XML sleep staging & arousal annotations
    stages, events_df = parse_profusion_xml(xml_path)
    if len(stages) == 0:
        raise ValueError("XML parsing failed or returned no stages.")

    # 2. Load EDF recording into memory
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # 3. Standardise montage to match standard clinical derivations
    raw = standardize_montage(raw)
    target_channels = list(DERIVATION_MAP.keys())
    existing = [ch for ch in target_channels if ch in raw.ch_names]
    
    if len(existing) < 3:
        raise ValueError(f"Missing required channels in {os.path.basename(edf_path)}. Found: {existing}")
    
    # 4. Isolate target signal channels
    raw.pick(existing)
    
    # 5. Apply bandpass filters
    eeg_eog_chs = [ch for ch in raw.ch_names if 'C4' in ch or 'E2' in ch]
    if eeg_eog_chs:
        raw.filter(0.3, 35.0, picks=eeg_eog_chs, fir_design='firwin', verbose=False)
        
    emg_chs = [ch for ch in raw.ch_names if 'Chin' in ch]
    if emg_chs:
        raw.filter(10.0, 45.0, picks=emg_chs, fir_design='firwin', verbose=False)
    
    # 6. Resample signals to target frequency
    raw.resample(FS) 
    data = raw.get_data()  # Shape layout: (channels=3, total_time_samples)
    
    # Clean workspace memory
    try:
        raw.close()
    except Exception:
        pass
    del raw 

    # 7. Synchronise signal bounds with annotation epoch limits
    n_epochs_edf = data.shape[1] // SAMPLES
    n_epochs_xml = len(stages)
    common_epochs = min(n_epochs_edf, n_epochs_xml)
    
    data = data[:, :common_epochs * SAMPLES]  # Shape: (3, common_epochs * 3000)
    stages = stages[:common_epochs]           # Shape: (common_epochs,)
    
    # 8. Construct 10Hz target arousal mask
    y_arousal = create_arousal_mask(events_df, common_epochs, target_hz=10) # Shape: (common_epochs, 300)

    # 9. Perform outlier clipping and z-score normalisation per channel
    for i in range(data.shape[0]):
        limit = np.percentile(np.abs(data[i]), 95) * 5
        data[i] = np.clip(data[i], -limit, limit)
        data[i] = (data[i] - np.mean(data[i])) / (np.std(data[i]) + 1e-6)

    # 10. Reshape array into individual epoch tensors
    # Initial shape: (3_channels, common_epochs, 3000_samples)
    x = data.reshape(3, common_epochs, SAMPLES)
    
    # Transpose layout to PyTorch standard: (num_epochs, channels=3, samples=3000)
    x = np.transpose(x, (1, 0, 2)) 
    
    return {
        'x': torch.from_numpy(x).float(),
        'y_stage': torch.from_numpy(stages).long(),
        'y_arousal': torch.from_numpy(y_arousal).float()
    }


# =====================================================================
# BATCH EXECUTION ORCHESTRATOR
# =====================================================================

def main():
    edf_files = glob.glob(os.path.join(EDF_PATH, "*.edf"))

    if not edf_files:
        print(f"No EDF files found in {EDF_PATH}. Check your path!")
        return

    for f in edf_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        save_name = f"{base_name}.pt"
        full_save_path = os.path.join(SAVE_PATH, save_name)

        # 1. Skip previously processed records
        if os.path.exists(full_save_path):
            print(f"⏩ Skipping: {base_name} (Already exists in {SAVE_PATH})")
            continue

        print(f"Processing: {base_name}...")
        
        # 2. Locate corresponding XML annotation file
        xml_path = os.path.join(LABEL_PATH, f"{base_name}.edf.xml")
        if not os.path.exists(xml_path):
            xml_path = os.path.join(LABEL_PATH, f"{base_name}.scoredata.xml")
            
        if not os.path.exists(xml_path):
            print(f"    ⚠️ Skipping: Could not find XML file for {base_name}")
            continue

        # 3. Run processing pipeline and save PyTorch dictionary
        try:
            processed_data_dict = preprocess_study(f, xml_path)
            
            torch.save(processed_data_dict, full_save_path)
            
            n_saved = processed_data_dict['x'].shape[0]
            print(f"    ✅ Successfully synced and saved {n_saved} epochs.")
            
            del processed_data_dict  # Free allocated RAM
            
        except Exception as e:
            print(f"    ❌ Failed {f}: {e}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
