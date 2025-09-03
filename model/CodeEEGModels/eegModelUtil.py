import numpy as np
import os
import sys
import mne
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from mne.preprocessing import ICA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

mne.set_log_level('ERROR')

#SAMPLING RATE = 500
NUM_SUBJECTS = 2
NUM_TRIALS = 514
NUM_CHANNELS = 24
NUM_TIMEPOINTS = 1001
CLASSES = ['Car4', 'Air4']
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 16
T_BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 10
EVALUATION_TYPE = "leave_one_subject_out"  # Global variable for evaluation type
TEST_SUBJECT = 0
TIME_FREQUENCY = False

class EEGDataset(Dataset):
    def __init__(self, data, labels, mean=None, std=None):
        self.data = torch.stack(data)  # shape: (N, 24, 1001)
        self.labels = torch.tensor(labels)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].float()
        y = self.labels[idx]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 24, 1001]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[None, :, None]) / self.std[None, :, None]
        return x, y

def set_parameter(para, value):
    if para == 'NUM_SUBJECTS':
        global NUM_SUBJECTS
        NUM_SUBJECTS = value
    elif para == 'NUM_TRIALS':
        global NUM_TRIALS
        NUM_TRIALS = value
    elif para == 'NUM_CHANNELS':
        global NUM_CHANNELS
        NUM_CHANNELS = value
    elif para == 'BATCH_SIZE':
        global BATCH_SIZE
        BATCH_SIZE = value
        global T_BATCH_SIZE 
        T_BATCH_SIZE = value
    elif para =='LR':
        global LR
        LR = value
    elif para == 'NUM_EPOCHS':
        global NUM_EPOCHS
        NUM_EPOCHS = value
        
def get_parameters():
    return NUM_SUBJECTS, NUM_TRIALS, NUM_CHANNELS, NUM_TIMEPOINTS, CLASSES, NUM_CLASSES, BATCH_SIZE, T_BATCH_SIZE, LR, NUM_EPOCHS, EVALUATION_TYPE, TEST_SUBJECT

def preprocess_eeg_data():
    """
    Returns (batch size x channel x freq x time), batch size x label
    """
    all_data = []
    all_labels = []
    
    event_id = {
        '1998': 1, '1999': 2, '200': 3, 'Air1': 4, 'Air2': 5, 'Air3': 6, 'Air4': 7, 'AirExtra': 8, 
        'Car1': 9, 'Car2': 10, 'Car3': 11, 'Car4': 12, 'OVTK_StimulationId_ExperimentStop': 13, 
        'OVTK_StimulationId_Label_3C': 14, 'Vib1': 15, 'Vib2': 16, 'Vib3': 17, 'Vib4': 18, 'VibExtra': 19
    }
    
    desired_codes = [event_id[c] for c in CLASSES]
    code_to_label = {code: idx for idx, code in enumerate(sorted(desired_codes))}

    for subject_idx in range(1, NUM_SUBJECTS + 1):  # sub-01 to sub-34
        subject_id = f"sub-{subject_idx:02d}"
        try:
            X, y, _, num_trials = load_eeg_epochs(subject_id=subject_id) 
        except Exception as e:
            print(f"Skipping {subject_id} due to error: {e}")
            traceback.print_exc()
        for trial in range(num_trials):
            # X[i]: EEG trial i, shape: [24, 1001] (24 channels, 1001 timepoints
            # Y[i]: Numeric label for trial i (plugging in y[i] into event_id tells us which stimulus was presented))
            # Convert each trial to a tensor: shape [channels, time]
            trial_label = y[trial]
            if trial_label not in desired_codes:
                continue  # skip trials with unwanted labels
            trial_tensor = torch.tensor(X[trial])  # shape: [24, 1001]
            
            #augment
            trial_tensor = augment_eeg_tensor(trial_tensor)
            # 🔑 Transform to time–frequency domain
            if TIME_FREQUENCY:
                trial_tensor = time_frequency_transform(trial_tensor)  # shape (channels, freq, time_steps)
            
            all_data.append(trial_tensor) # contains 34 subjects x 516 segments
            all_labels.append(code_to_label[trial_label])
    return all_data, all_labels

def get_cv_split(X, y, n_splits=5, fold=0, seed=42, evaluation_type="within_subject"):
    """
    Splits the EEG dataset for cross-validation.
    
    Args:
        X (list or np.array): List of EEG trials, each (24, 1001) shape.
        y (list or np.array): Corresponding labels.
        n_splits (int): Number of CV folds.
        fold (int): Which fold to use for testing (0-indexed).
        seed (int): Random seed for reproducibility.
        evaluation_type (str): "within_subject" or "leave_one_subject_out"

    Returns:
        X_train, X_test, y_train, y_test: numpy arrays.
    """
    
    if evaluation_type == "within_subject":
        # Current implementation: random trial split across all subjects
        X = np.stack([x.numpy() if isinstance(x, torch.Tensor) else x for x in X])
        y = np.array(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
        train_idx, test_idx = splits[fold]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    elif evaluation_type == "leave_one_subject_out":
        # LOSO: train on N-1 subjects, test on 1 subject
        return get_loso_split(X, y, fold, seed)
    
    else:
        raise ValueError(f"Unknown evaluation_type: {evaluation_type}. Use 'within_subject' or 'leave_one_subject_out'")

def get_loso_split(X, y, test_subject_idx=0, seed=42):
    """
    Leave-One-Subject-Out (LOSO) cross-validation.
    
    Args:
        X (list): List of EEG trials
        y (list): Corresponding labels
        test_subject_idx (int): Which subject to use for testing (0-indexed)
        seed (int): Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test: numpy arrays.
    """
    total_trials = len(X)
    
    # For LOSO, we need to estimate how many subjects we have
    # Since we don't know the exact trials per subject, we'll use a more flexible approach
    # Assuming roughly equal distribution across subjects
    estimated_trials_per_subject = total_trials // NUM_SUBJECTS
    num_subjects = min(NUM_SUBJECTS, total_trials // max(1, estimated_trials_per_subject))
    
    if test_subject_idx >= num_subjects:
        raise ValueError(f"test_subject_idx {test_subject_idx} >= num_subjects {num_subjects} (total_trials: {total_trials})")
    
    # Calculate start and end indices for test subject
    trials_per_subject = total_trials // num_subjects
    test_start = test_subject_idx * trials_per_subject
    test_end = min(test_start + trials_per_subject, total_trials) if test_subject_idx < num_subjects - 1 else total_trials
    
    # Split data
    X = np.stack([x.numpy() if isinstance(x, torch.Tensor) else x for x in X])
    y = np.array(y)
    
    # Test set: one subject
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]
    
    # Train set: all other subjects
    X_train = np.concatenate([X[:test_start], X[test_end:]])
    y_train = np.concatenate([y[:test_start], y[test_end:]])
    
    print(f"LOSO Split: Test Subject {test_subject_idx + 1}, "
          f"Train: {len(X_train)} trials, Test: {len(X_test)} trials")
    
    return X_train, X_test, y_train, y_test

def load_eeg_epochs(subject_id="sub-01", condition="E", tmin=0.0, tmax=1.0,
                    data_dir=r"tactile files\touch_data\NeurosenseDB", reject_threshold=85):
    """
    Load and preprocess EEG data from an EDF file for a given subject and condition.
    Replaces OpenViBE annotation codes with human-readable stimulus labels.

    Returns:
        X: EEG epoch data (numpy array)
        y: Numeric labels per epoch
        event_id: Dictionary mapping stimulus names to integer event codes
    """

    # Construct EDF path
    edf_path = os.path.join(data_dir, subject_id, "eeg", f"{subject_id}_{condition}.edf")
    # print(f"Loading file: {edf_path}")

    # Load raw EEG data
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Remove gyro channels if present
    gyro_channels = ['Gyro 1', 'Gyro 2', 'Gyro 3']
    raw.drop_channels([ch for ch in gyro_channels if ch in raw.ch_names])

    # Rename EEG channels if count matches expected
    new_labels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
                  'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','M1',
                  'M2','AFz','CPz','POz']
    if len(raw.ch_names) == len(new_labels):
        raw.rename_channels(dict(zip(raw.ch_names, new_labels)))
    else:
        print("Warning: Channel count mismatch. Skipping channel renaming.")

    # --- ICA Cleaning Step ---
    raw, ica = run_ica_cleaning(raw, subject_id=subject_id)
    if ica is not None:
        print(f"Removed ICs for {subject_id}: {ica.exclude}") 
    else:
        print("ICA failed.")
    
    # Define mapping from annotation label to stimulus name
    label_to_stimulus = {
        'OVTK_StimulationId_Label_00': 'Car1',
        'OVTK_StimulationId_Label_01': 'Car2',
        'OVTK_StimulationId_Label_02': 'Car3',
        'OVTK_StimulationId_Label_03': 'Car4',
        'OVTK_StimulationId_Label_04': 'Air1',
        'OVTK_StimulationId_Label_05': 'Air2',
        'OVTK_StimulationId_Label_06': 'Air3',
        'OVTK_StimulationId_Label_07': 'Air4',
        'OVTK_StimulationId_Label_08': 'Vib1',
        'OVTK_StimulationId_Label_09': 'Vib2',
        'OVTK_StimulationId_Label_0A': 'Vib3',
        'OVTK_StimulationId_Label_0B': 'Vib4',
        'OVTK_StimulationId_Label_12': 'VibExtra',
        'OVTK_StimulationId_Label_13': 'AirExtra'
    }

    # Apply label mapping to annotations
    mapped_descriptions = [
        label_to_stimulus.get(desc, desc) for desc in raw.annotations.description
    ]
    raw.set_annotations(mne.Annotations(
        onset=raw.annotations.onset,
        duration=raw.annotations.duration,
        description=mapped_descriptions
    ))

    # Bandpass filter
    raw.filter(1., 40., fir_design='firwin')

    # Convert annotations to events using string stimulus labels
    events, event_id = mne.events_from_annotations(raw)

      # Epoch extraction with rejection of bad epochs (peak-to-peak > threshold)
    reject_criteria = dict(eeg=reject_threshold)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, reject=reject_criteria, preload=True, event_repeated='merge')
    
    # print(f"{subject_id}: {len(epochs)} epochs after rejection")

    # Prepare data
    X = epochs.get_data() # (n_trials, n_channels, n_timepoints)
    y = epochs.events[:, 2]  # array (n_trials): each element in y is an integer representing stimulus label for the corresponding trial in X
    #event id is a dictionary mapping stimulus names to integer event codes
    
    return zscore_normalize(X), y, event_id, len(epochs)

def run_ica_cleaning(raw, 
                     subject_id=None,
                     n_components=20, 
                     method="picard", 
                     random_state=97,
                     frontal_chs=['Fp1', 'Fp2', 'AFz', 'Fz', 'F3', 'F4'],
                     ica_dir="ica_weights"):
    """
    Run ICA on EEG data and remove common artifacts (eye blinks, ECG, muscle).
    Uses frontal EEG channels as proxy EOG and automatic detection for muscle artifacts.
    Saves and loads ICA weights to avoid recomputation.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data (preloaded)
    subject_id : str
        Subject identifier, used for saving/loading ICA weights
    n_components : float or int
        Number of ICA components or explained variance
    method : str
        ICA method (picard)
    random_state : int
        Random seed
    frontal_chs : list of str
        Frontal channels used as proxy EOG
    ica_dir : str
        Directory to save ICA weights

    Returns
    -------
    clean_raw : mne.io.Raw
        EEG data with artifact ICs removed
    ica : ICA object or None
        Fitted ICA object, or None if ICA failed
    """
    os.makedirs(ica_dir, exist_ok=True)
    ica_fname = os.path.join(ica_dir, f"{subject_id}-ica.fif") if subject_id else None

    # --- Try to load existing ICA ---
    if ica_fname and os.path.exists(ica_fname):
        print(f"Loading existing ICA for {subject_id}")
        ica = mne.preprocessing.read_ica(ica_fname)
        try:
            clean_raw = ica.apply(raw.copy())
            return clean_raw, ica
        except Exception as e:
            print(f"Applying saved ICA failed: {e}")
            traceback.print_exc()

    # --- Run ICA if not saved or failed to apply ---
    try:
        # --- Ensure montage ---
        try:
            raw.get_montage()
        except:
            raw.set_montage("standard_1020")

        # --- Prep data ---
        raw_filt = raw.copy().filter(1., 40., fir_design='firwin')

        # --- Fit ICA ---
        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        ica.fit(raw_filt)

        # --- Detect EOG artifacts using frontal channels ---
        artifact_inds = []
        frontal_chs_exist = [ch for ch in frontal_chs if ch in raw.ch_names]
        if frontal_chs_exist:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=frontal_chs_exist)
            artifact_inds.extend(eog_inds)

        # --- Detect ECG artifacts ---
        ecg_chs = [ch for ch in raw.ch_names if 'ECG' in ch.upper()]
        if ecg_chs:
            try:
                ecg_inds, _ = ica.find_bads_ecg(raw)
                artifact_inds.extend(ecg_inds)
            except Exception:
                pass

        # --- Detect muscle artifacts ---
        try:
            muscle_inds, _ = ica.find_bads_muscle(raw, threshold=4.0)
            artifact_inds.extend(muscle_inds)
        except Exception:
            pass

        # --- Remove duplicates ---
        ica.exclude = list(set(artifact_inds))

        # --- Apply ICA ---
        clean_raw = ica.apply(raw.copy())

        # --- Save ICA for future use ---
        if ica_fname:
            ica.save(ica_fname)
            print(f"Saved ICA for {subject_id} to {ica_fname}")

        return clean_raw, ica

    except Exception as e:
        print(f"ICA failed for this subject: {e}")
        traceback.print_exc()
        return raw, None

def augment_eeg_tensor(trial_tensor, noise_std=0.01, time_shift_max=10, scale_range=(0.9, 1.1), augment_prob=0.5):
    """
    Augment a single EEG trial tensor with noise, time shift, and scaling.
    trial_tensor: torch tensor, shape [channels, timepoints]
    Returns augmented tensor.
    """
    X_aug = trial_tensor.clone()

    if np.random.rand() < augment_prob:
        noise = torch.randn_like(X_aug) * noise_std
        X_aug = X_aug + noise

    if np.random.rand() < augment_prob:
        shift = np.random.randint(-time_shift_max, time_shift_max + 1)
        X_aug = torch.roll(X_aug, shifts=shift, dims=1)

    if np.random.rand() < augment_prob:
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        X_aug = X_aug * scale_factor

    return X_aug

def time_frequency_transform(x, n_fft=128, hop_length=64):
    """
    x: Tensor of shape (channels, time)
    Returns: (channels, freq, time_steps) magnitude spectrogram
    """
    # Single trial -> (channels, time)
    channels, time = x.shape
    out = []
    for ch in range(channels):
        stft_ch = torch.stft(
            x[ch, :],
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        )
        out.append(stft_ch.abs().unsqueeze(0))  # (1, freq, time_steps)
    return torch.cat(out, dim=0)  # (channels, freq, time_steps)

def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == 'cuda':
        print('Your GPU device name :', torch.cuda.get_device_name())
    return dev

def load_data(train_data, train_labels, test_data, test_labels, val_split=0.1):

    # Convert to tensors
    train_data_tensors = [torch.tensor(x, dtype=torch.float32) for x in train_data]
    test_data_tensors = [torch.tensor(x, dtype=torch.float32) for x in test_data]

    # Split training into train + val
    train_X, val_X, train_y, val_y = train_test_split(train_data_tensors, train_labels, test_size=val_split, stratify=train_labels, random_state=42)

    train_tensor = torch.stack(train_X)
    mean = train_tensor.mean(dim=(0, 2))
    std = train_tensor.std(dim=(0, 2))

    train_dataset = EEGDataset(train_X, train_y, mean, std)
    val_dataset = EEGDataset(val_X, val_y, mean, std)
    test_dataset = EEGDataset(test_data_tensors, test_labels, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=T_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=T_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=NUM_EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    total_steps = epochs * len(train_loader)  # for tqdm
    progress_bar = tqdm(total=total_steps, desc="Training", ncols=100)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device).float(), labels.to(device).long()
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    progress_bar.close()
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device).float(), labels.to(device).long()
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return all_preds, all_labels

def zscore_normalize(X):
    X_norm = np.zeros_like(X)
    for i in range(X.shape[0]):  # For each trial
        X_norm[i] = (X[i] - np.mean(X[i], axis=1, keepdims=True)) / np.std(X[i], axis=1, keepdims=True)
    return X_norm

def plot_training_curves(train_losses, val_losses, model_name="EEGTCNet", save_path=None, evaluation_type=None, classes=None, lr=None, epochs=None):
    """
    Plot training and validation loss curves with model information.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
        evaluation_type (str, optional): Type of evaluation (within_subject/loso)
        classes (list, optional): List of classes being classified
        lr (float, optional): Learning rate used
        epochs (int, optional): Number of epochs
    """
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Create figure with larger size for more information
    plt.figure(figsize=(12, 8))
    
    # Plot training curves
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Create title with model information
    title = f'{model_name} Training Curves'
    if classes:
        title += f'\nClasses: {", ".join(classes)}'
    if evaluation_type:
        title += f' | Evaluation: {evaluation_type.upper()}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add parameter information in a text box
    param_text = f"Learning Rate: {lr}\nEpochs: {epochs}\nFinal Train Loss: {train_losses[-1]:.4f}\nFinal Val Loss: {val_losses[-1]:.4f}"
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Add loss difference plot
    plt.subplot(2, 1, 2)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(epochs_range, loss_diff, 'g-', label='|Train - Val| Loss', linewidth=2)
    plt.title('Loss Difference (Overfitting Indicator)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Difference', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def output_confusion_matrix(preds, labels, classes, model_filename):
    """
    Takes in predictions and labels to print a confusion matrix
    and save a heatmap image to 'model/CodeEEGModels/heatmaps/'.
    
    Args:
        preds: List or array of predicted labels.
        labels: List or array of true labels.
        classes: List of class names.
        model_filename: String filename of the model (e.g., 'EEGNet.py').
    """
    
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("Confusion matrix: ")
    print(cm_df)
    
    # Directory path
    save_dir = os.path.join("model", "CodeEEGModels", "heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    
    # Timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_filename}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    # Plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_filename}')
    
    # Save and close
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {save_path}")


