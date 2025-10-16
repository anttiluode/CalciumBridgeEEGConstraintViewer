"""
Calcium-Bridged Temporal EEG Decoder (V2 - Extended Time Window)
Integrates Phase-Calcium-Latent constraint satisfaction dynamics with EEG temporal windows.

Core Concept: Each EEG time window is processed by a constraint solver whose 
calcium/W state carries over to initialize the next window, modeling how the brain 
sequentially satisfies perceptual constraints.

V2 Update: The time window has been extended to 550ms based on ERP analysis from the 
Alljoined1 paper, adding a 'CognitiveEvaluation' stage to capture late-stage
semantic and working memory signals.
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import threading
import queue
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from datasets import load_dataset
    torch.backends.cudnn.benchmark = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512
BATCH_SIZE = 64

# --- CRITICAL CHANGE V2 ---
# Extended temporal windows to capture later cognitive processing (P300/N400/P600)
# This aligns the model's "attention span" with the neuroscience data.
TIME_WINDOWS = [
    (50, 150, "EarlyVisual"),      # Low-level visual constraints (P100)
    (150, 250, "MidFeature"),      # Mid-level binding (N170/P200)
    (250, 350, "LateSemantic"),    # High-level semantics (P300/N400 start)
    (350, 550, "CognitiveEvaluation") # Deeper context, memory, final check (P300/P600)
]

TARGET_CATEGORIES = {
    'elephant': 22, 'giraffe': 25, 'bear': 23, 'zebra': 24,
    'cow': 21, 'sheep': 20, 'horse': 19, 'dog': 18, 'cat': 17, 'bird': 16,
    'airplane': 5, 'train': 7, 'boat': 9, 'bus': 6, 'truck': 8,
    'motorcycle': 4, 'bicycle': 2, 'car': 3,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15,
}

CATEGORY_NAMES = {v: k for k, v in TARGET_CATEGORIES.items()}

class CalciumAttentionModule(nn.Module):
    """
    Phase-Calcium-Latent dynamics for one time window.
    Models constraint satisfaction via neuromorphic oscillator dynamics.
    """
    def __init__(self, n_features, d_model=256):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Phase dynamics (Kuramoto-like)
        self.phase_proj = nn.Linear(n_features, d_model)
        
        # Calcium dynamics (gating/attention)
        self.ca_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Sigmoid()
        )
        
        # Latent coupling matrix (W) - learned constraint structure
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, prev_ca=None, prev_W=None):
        """
        x: Input features [batch, n_features]
        prev_ca: Previous window's calcium state [batch, d_model]
        prev_W: Previous window's coupling matrix [d_model, d_model]
        
        Returns: features, calcium_state, W_matrix
        """
        batch_size = x.size(0)
        
        # Phase projection
        phi = self.phase_proj(x)  # [batch, d_model]
        
        # Initialize or carry over calcium
        if prev_ca is None:
            ca = torch.zeros(batch_size, self.d_model, device=x.device)
        else:
            ca = prev_ca.clone()
        
        # Initialize or carry over W (coupling structure)
        W = self.W if prev_W is None else prev_W
        
        # Calcium accumulation (coherence-based)
        # High when features are aligned (low when conflicting)
        coherence = torch.abs(torch.cos(phi[:, :, None] - phi[:, None, :]))
        ca_update = torch.mean(coherence, dim=2)  # [batch, d_model]
        ca = ca * 0.95 + ca_update * 0.05  # Temporal integration
        
        # Calcium-gated attention
        ca_gate = self.ca_gate(ca)  # [batch, d_model//2]
        
        # Apply constraint coupling (W matrix)
        # This is where "mutual constraint satisfaction" happens
        coupled = torch.matmul(phi, W)  # [batch, d_model]
        
        # Gate the coupling by calcium (only attend where calcium is high)
        ca_gate_full = torch.cat([ca_gate, ca_gate], dim=1)  # Expand to d_model
        features = coupled * ca_gate_full
        
        # Normalize
        features = self.norm(features + phi)  # Residual connection
        
        return features, ca, W


class TemporalConstraintEEGModel(nn.Module):
    """
    Sequential constraint satisfaction across EEG time windows.
    Each window is a constraint solver whose state primes the next.
    (Dynamically sized based on TIME_WINDOWS constant)
    """
    def __init__(self, n_channels=64, num_classes=len(TARGET_CATEGORIES)):
        super().__init__()
        self.n_channels = n_channels
        
        # CNN feature extractors for each time window
        self.window_encoders = nn.ModuleList([
            self._build_cnn_encoder() for _ in TIME_WINDOWS
        ])
        
        # Calcium-attention modules for each window
        self.ca_modules = nn.ModuleList([
            CalciumAttentionModule(256, d_model=256) for _ in TIME_WINDOWS
        ])
        
        # --- CRITICAL CHANGE V2 ---
        # The input layer is now automatically larger (256 * 4) because len(TIME_WINDOWS) is 4.
        self.classifier = nn.Sequential(
            nn.Linear(256 * len(TIME_WINDOWS), 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def _build_cnn_encoder(self):
        """Simple CNN for one time window"""
        return nn.Sequential(
            nn.Conv1d(self.n_channels, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, eeg_windows):
        """
        eeg_windows: List of tensors [batch, channels, timepoints] for each window
        
        Returns: logits, calcium_states (for visualization/analysis)
        """
        batch_size = eeg_windows[0].size(0)
        
        # Process windows sequentially with calcium/W carryover
        window_features = []
        ca_state = None
        W_state = None
        ca_history = []
        
        for i, (encoder, ca_module, eeg_window) in enumerate(
            zip(self.window_encoders, self.ca_modules, eeg_windows)
        ):
            # Extract CNN features
            cnn_features = encoder(eeg_window).squeeze(-1)  # [batch, 256]
            
            # Apply constraint satisfaction dynamics
            features, ca_state, W_state = ca_module(
                cnn_features, 
                prev_ca=ca_state, 
                prev_W=W_state
            )
            
            window_features.append(features)
            ca_history.append(ca_state.detach().cpu().numpy())
        
        # Concatenate all window features
        combined = torch.cat(window_features, dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits, ca_history


class CalciumEEGDataset(Dataset):
    """Dataset that provides EEG data split by time windows"""
    def __init__(self, coco_path, annotations_path, split='train', 
                 max_samples=None, trials_to_average=1):
        self.coco_path = Path(coco_path)
        
        # Load dataset
        print(f"Loading Alljoined ({split})...")
        self.dataset = load_dataset("Alljoined/05_125", split=split, streaming=False)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(int(max_samples), len(self.dataset))))
        
        # Load COCO annotations
        print(f"Loading COCO annotations...")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        self.image_categories = defaultdict(set)
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if ann['category_id'] in CATEGORY_NAMES:
                self.image_categories[img_id].add(ann['category_id'])
        
        # Pre-cache samples
        print("Pre-caching EEG data...")
        self.samples = []
        for idx, sample in enumerate(self.dataset):
            coco_id = sample['coco_id']
            if coco_id in self.image_categories and len(self.image_categories[coco_id]) > 0:
                label = torch.zeros(len(TARGET_CATEGORIES))
                for cat_id in self.image_categories[coco_id]:
                    if cat_id in CATEGORY_NAMES:
                        cat_idx = list(TARGET_CATEGORIES.values()).index(cat_id)
                        label[cat_idx] = 1.0
                
                if label.sum() > 0:
                    self.samples.append((idx, label))
        
        print(f"Cached {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_idx, label = self.samples[idx]
        sample = self.dataset[sample_idx]
        
        eeg_data = np.array(sample['EEG'], dtype=np.float32)
        
        # Extract time windows
        eeg_windows = []
        for start_ms, end_ms, _ in TIME_WINDOWS:
            start_idx = int((start_ms / 1000.0) * EEG_SAMPLE_RATE)
            end_idx = int((end_ms / 1000.0) * EEG_SAMPLE_RATE)
            
            if eeg_data.shape[1] >= end_idx:
                window = eeg_data[:, start_idx:end_idx]
            else:
                window = eeg_data[:, start_idx:]
                # Pad if needed
                if window.shape[1] < (end_idx - start_idx):
                    pad_width = (end_idx - start_idx) - window.shape[1]
                    window = np.pad(window, ((0,0), (0, pad_width)), mode='edge')
            
            # Normalize
            window = (window - window.mean(axis=1, keepdims=True)) / \
                     (window.std(axis=1, keepdims=True) + 1e-8)
            
            eeg_windows.append(torch.from_numpy(window).float())
        
        return eeg_windows, label


class CalciumEEGTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calcium-Bridged Temporal EEG Decoder V2")
        self.geometry("1200x850")
        
        self.coco_path = ""
        self.annotations_path = ""
        self.train_thread = None
        self.stop_flag = threading.Event()
        self.log_queue = queue.Queue()
        
        self.setup_gui()
        self.process_logs()
    
    def setup_gui(self):
        # Title
        title = tk.Label(self, text="Calcium-Bridged Temporal EEG Decoder (V2 - Extended Window)", 
                        font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        info = tk.Label(self, 
                       text="Sequential constraint satisfaction across 4 ERP time windows up to 550ms\n"
                            "Calcium/W state from early windows primes later windows",
                       fg="blue", font=("Arial", 9))
        info.pack(pady=5)
        
        # Paths
        path_frame = ttk.LabelFrame(self, text="Dataset")
        path_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(path_frame, text="COCO:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.coco_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.coco_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_coco).grid(row=0, column=2)
        
        tk.Label(path_frame, text="Annotations:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.ann_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.ann_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_ann).grid(row=1, column=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(self, text="Training Settings")
        settings_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(settings_frame, text="Max Samples:").grid(row=0, column=0, padx=5)
        self.max_var = tk.IntVar(value=3000)
        tk.Spinbox(settings_frame, from_=1000, to=10000, increment=1000,
                  textvariable=self.max_var, width=10).grid(row=0, column=1)
        
        tk.Label(settings_frame, text="Epochs:").grid(row=0, column=2, padx=5)
        self.epochs_var = tk.IntVar(value=100)
        tk.Spinbox(settings_frame, from_=50, to=500, increment=50,
                  textvariable=self.epochs_var, width=10).grid(row=0, column=3)
        
        # --- CRITICAL CHANGE V2 ---
        # Updated GUI to reflect the new 4-stage process.
        windows_frame = ttk.LabelFrame(self, text="Constraint Satisfaction Stages")
        windows_frame.pack(pady=5, padx=10, fill=tk.X)
        
        for start, end, label in TIME_WINDOWS:
            desc = {
                "EarlyVisual": "Low-level visual features (edges, textures)",
                "MidFeature": "Mid-level binding (parts, shapes)",
                "LateSemantic": "High-level semantics (concepts, context)",
                "CognitiveEvaluation": "Memory, context check, final decision"
            }
            tk.Label(windows_frame, 
                    text=f"{label} ({start}-{end}ms): {desc[label]}", 
                    font=("Courier", 9)).pack(anchor=tk.W, padx=10, pady=2)
        
        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        
        self.train_btn = tk.Button(btn_frame, text="Train Extended Model (V2)", 
                                   command=self.start_train,
                                   bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", 
                                  command=self.stop_train,
                                  bg="#f44336", fg="white",
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(self, text="Training Log")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=20, bg='black', fg='lightgreen',
                               font=('Courier', 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_coco(self):
        path = filedialog.askdirectory()
        if path:
            self.coco_var.set(path)
            self.coco_path = path
    
    def browse_ann(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.ann_var.set(path)
            self.annotations_path = path
    
    def log(self, msg):
        self.log_queue.put(msg)
    
    def process_logs(self):
        try:
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.process_logs)
    
    def start_train(self):
        if not self.coco_path or not self.annotations_path:
            messagebox.showerror("Error", "Select paths first")
            return
        
        self.stop_flag.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.train_thread = threading.Thread(target=self._train_model, daemon=True)
        self.train_thread.start()
    
    def stop_train(self):
        self.stop_flag.set()
    
    def _train_model(self):
        try:
            self.log("="*70)
            self.log("CALCIUM-BRIDGED TEMPORAL EEG DECODER (V2 - Extended Window)")
            self.log("="*70)
            self.log("\nConcept: Sequential constraint satisfaction across FOUR time windows")
            self.log("Now capturing late-stage cognitive evaluation signals up to 550ms\n")
            
            # Create dataset
            dataset = CalciumEEGDataset(
                self.coco_path,
                self.annotations_path,
                'train',
                self.max_var.get()
            )
            
            total = len(dataset)
            train_size = int(0.8 * total)
            val_size = total - train_size
            
            train_set, val_set = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.log(f"Train: {train_size}, Val: {val_size}")
            
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, 
                                    shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, 
                                   shuffle=False, num_workers=0, pin_memory=True)
            
            # Create model (will be automatically sized for 4 windows)
            model = TemporalConstraintEEGModel().to(DEVICE)
            self.log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            
            best_val_loss = float('inf')
            
            for epoch in range(self.epochs_var.get()):
                if self.stop_flag.is_set():
                    break
                
                # Train
                model.train()
                train_loss = 0
                for eeg_windows, labels in train_loader:
                    if self.stop_flag.is_set():
                        break
                    
                    eeg_windows = [w.to(DEVICE) for w in eeg_windows]
                    labels = labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    logits, _ = model(eeg_windows)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for eeg_windows, labels in val_loader:
                        eeg_windows = [w.to(DEVICE) for w in eeg_windows]
                        labels = labels.to(DEVICE)
                        logits, _ = model(eeg_windows)
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step()
                
                self.progress['value'] = ((epoch + 1) / self.epochs_var.get()) * 100
                
                if epoch % 5 == 0:
                    self.log(f"Epoch {epoch+1}/{self.epochs_var.get()}: "
                            f"TrLoss={train_loss:.4f} ValLoss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch
                    }, "calcium_bridge_eeg_model_v2.pth") # Save as V2
                    if epoch % 5 == 0:
                        self.log(f"  -> Saved V2 model (val_loss={val_loss:.4f})")
            
            self.log("\n" + "="*70)
            self.log("TRAINING COMPLETE")
            self.log(f"Best Val Loss: {best_val_loss:.4f}")
            self.log("="*70)
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)


if __name__ == "__main__":
    app = CalciumEEGTrainerGUI()
    app.mainloop()