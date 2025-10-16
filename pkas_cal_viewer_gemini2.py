"""
Calcium-Bridge EEG Constraint Viewer (V2.1 - Fixed)
Visualizes how constraint satisfaction unfolds across four temporal windows up to 550ms.

Shows:
1. Original COCO image
2. EEG heatmaps for each of the 4 time windows
3. Calcium "attention" evolution (what the model focuses on at each stage)
4. Top predictions crystallizing across the 4 windows

V2.1 Fixes:
- Corrected 'figsize' argument placement during figure creation.
- Corrected colorbar creation to use the figure object directly, resolving warnings.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path
from collections import defaultdict
import random

try:
    from datasets import load_dataset
except ImportError:
    print("Missing datasets library.")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512

TIME_WINDOWS = [
    (50, 150, "EarlyVisual"),
    (150, 250, "MidFeature"),
    (250, 350, "LateSemantic"),
    (350, 550, "CognitiveEvaluation")
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
TARGET_IDS = set(TARGET_CATEGORIES.values())
ALL_COCO_IDS = list(range(1, 91))
EXCLUDED_IDS = set(ALL_COCO_IDS) - TARGET_IDS


# === MODEL ARCHITECTURE (Must match V2 training code) ===

class CalciumAttentionModule(nn.Module):
    def __init__(self, n_features, d_model=256):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.phase_proj = nn.Linear(n_features, d_model)
        self.ca_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Sigmoid()
        )
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, prev_ca=None, prev_W=None):
        batch_size = x.size(0)
        phi = self.phase_proj(x)
        
        if prev_ca is None:
            ca = torch.zeros(batch_size, self.d_model, device=x.device)
        else:
            ca = prev_ca.clone()
        
        W = self.W if prev_W is None else prev_W
        
        coherence = torch.abs(torch.cos(phi[:, :, None] - phi[:, None, :]))
        ca_update = torch.mean(coherence, dim=2)
        ca = ca * 0.95 + ca_update * 0.05
        
        ca_gate = self.ca_gate(ca)
        coupled = torch.matmul(phi, W)
        ca_gate_full = torch.cat([ca_gate, ca_gate], dim=1)
        features = coupled * ca_gate_full
        features = self.norm(features + phi)
        
        return features, ca, W


class TemporalConstraintEEGModel(nn.Module):
    def __init__(self, n_channels=64, num_classes=len(TARGET_CATEGORIES)):
        super().__init__()
        self.n_channels = n_channels
        
        self.window_encoders = nn.ModuleList([
            self._build_cnn_encoder() for _ in TIME_WINDOWS
        ])
        
        self.ca_modules = nn.ModuleList([
            CalciumAttentionModule(256, d_model=256) for _ in TIME_WINDOWS
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * len(TIME_WINDOWS), 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def _build_cnn_encoder(self):
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
    
    def forward(self, eeg_windows, return_intermediates=False):
        batch_size = eeg_windows[0].size(0)
        
        window_features, ca_history, W_history, window_logits_list = [], [], [], []
        ca_state, W_state = None, None
        
        for i, (encoder, ca_module, eeg_window) in enumerate(
            zip(self.window_encoders, self.ca_modules, eeg_windows)
        ):
            cnn_features = encoder(eeg_window).squeeze(-1)
            features, ca_state, W_state = ca_module(cnn_features, ca_state, W_state)
            
            window_features.append(features)
            if return_intermediates:
                ca_history.append(ca_state.detach())
                W_history.append(W_state.detach())
                
                padded_features = window_features + [
                    torch.zeros_like(features) for _ in range(len(TIME_WINDOWS) - len(window_features))
                ]
                intermediate_logits = self.classifier(torch.cat(padded_features, dim=1))
                window_logits_list.append(intermediate_logits.detach())
        
        combined = torch.cat(window_features, dim=1)
        logits = self.classifier(combined)
        
        if return_intermediates:
            return logits, ca_history, W_history, window_logits_list
        return logits, ca_history


# === DATA LOADER ===
class FilteredTestDataset:
    def __init__(self, annotations_path, max_samples=1000):
        print("Loading and filtering test dataset...")
        self.eeg_dataset = load_dataset("Alljoined/05_125", split='test', streaming=False).select(range(max_samples))
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        image_annotations = defaultdict(set)
        for ann in coco_data['annotations']:
            image_annotations[ann['image_id']].add(ann['category_id'])

        self.filtered_samples = []
        for idx, sample in enumerate(self.eeg_dataset):
            ann_ids = image_annotations.get(sample['coco_id'], set())
            if not any(cat_id in EXCLUDED_IDS for cat_id in ann_ids) and any(cat_id in TARGET_IDS for cat_id in ann_ids):
                self.filtered_samples.append({
                    'coco_id': sample['coco_id'],
                    'eeg_data': np.array(sample['EEG'], dtype=np.float32)
                })
        print(f"Loaded {len(self.filtered_samples)} filtered test samples.")
        if not self.filtered_samples: raise RuntimeError("No suitable test samples found.")

    def get_eeg_windows(self, sample_info):
        eeg_data = sample_info['eeg_data']
        eeg_windows = []
        for start_ms, end_ms, _ in TIME_WINDOWS:
            start_idx, end_idx = int(start_ms / 1000 * EEG_SAMPLE_RATE), int(end_ms / 1000 * EEG_SAMPLE_RATE)
            n_timepoints = end_idx - start_idx
            window = eeg_data[:, start_idx:end_idx] if eeg_data.shape[1] >= end_idx else eeg_data[:, start_idx:]
            
            if window.shape[1] != n_timepoints:
                pad_width = n_timepoints - window.shape[1]
                window = np.pad(window, ((0,0), (0, pad_width)), 'edge') if pad_width > 0 else window[:, :n_timepoints]
            
            window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
            eeg_windows.append(window)
        return eeg_windows

    def get_random_sample_info(self):
        return random.choice(self.filtered_samples)

# === VIEWER APPLICATION ===
class CalciumBridgeViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calcium-Bridge EEG Constraint Viewer V2 (Extended Window)")
        self.geometry("2000x1000")
        self.model, self.test_data = None, None
        self.setup_gui()

    def setup_gui(self):
        control_frame = ttk.Frame(self); control_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Label(control_frame, text="COCO Path:").pack(side=tk.LEFT, padx=5)
        self.coco_var = tk.StringVar(); ttk.Entry(control_frame, textvariable=self.coco_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Browse", command=self.browse_coco).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Annotations:").pack(side=tk.LEFT, padx=5)
        self.ann_var = tk.StringVar(); ttk.Entry(control_frame, textvariable=self.ann_var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Browse", command=self.browse_ann).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load V2 Model", command=self.load_model).pack(side=tk.LEFT, padx=20)
        self.test_btn = ttk.Button(control_frame, text="Test Random Sample", command=self.test_sample, state=tk.DISABLED); self.test_btn.pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(control_frame, text="Model: Not loaded", fg="gray"); self.status_label.pack(side=tk.LEFT, padx=20)

        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL); main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        image_frame = ttk.Frame(main_paned, width=400); main_paned.add(image_frame, weight=0)
        ttk.Label(image_frame, text="COCO Image", font=("Arial", 12, "bold")).pack(pady=5)
        self.image_canvas = tk.Canvas(image_frame, width=400, height=400, bg='lightgray'); self.image_canvas.pack()
        self.coco_id_label = ttk.Label(image_frame, text="COCO ID: N/A"); self.coco_id_label.pack(pady=5)
        
        self.notebook = ttk.Notebook(main_paned); main_paned.add(self.notebook, weight=1)
        self.create_tabs()

    def create_tabs(self):
        self.constraint_fig, self.constraint_canvas = self.create_tab("Constraint Satisfaction", "How predictions crystallize as constraints are satisfied")
        self.calcium_fig, self.calcium_canvas = self.create_tab("Calcium Attention", "Calcium state evolution: What the model 'focuses on' at each stage")
        self.eeg_fig, self.eeg_canvas = self.create_tab("EEG Heatmaps", "Raw EEG signals for each time window")

    def create_tab(self, title, description):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text=title)
        ttk.Label(tab, text=description, font=("Arial", 11)).pack(pady=5)
        fig = plt.Figure()
        canvas = FigureCanvasTkAgg(fig, tab); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return fig, canvas

    def browse_coco(self):
        path = filedialog.askdirectory(); self.coco_var.set(path); self.coco_path = path
    
    def browse_ann(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")]); self.ann_var.set(path); self.annotations_path = path

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")], title="Select calcium_bridge_eeg_model_v2.pth")
        if not model_path or not self.annotations_path: return
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            self.model = TemporalConstraintEEGModel().to(DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.test_data = FilteredTestDataset(self.annotations_path)
            self.status_label.config(text="Model: V2 Loaded ✓", fg="green")
            self.test_btn.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Error", f"Failed to load model:\n{e}"); print(traceback.format_exc())

    def _fetch_image(self, coco_id):
        formatted_id = f"{coco_id:012d}.jpg"
        for s in ["train2017", "val2017", "test2017"]:
            path = os.path.join(self.coco_path, s, formatted_id)
            if os.path.exists(path): return Image.open(path).convert("RGB")
        return None

    def test_sample(self):
        if not self.model: return
        try:
            sample_info = self.test_data.get_random_sample_info()
            image = self._fetch_image(sample_info['coco_id'])
            if image: self.display_image(image, sample_info['coco_id'])
            
            eeg_windows_np = self.test_data.get_eeg_windows(sample_info)
            eeg_windows = [torch.from_numpy(w).unsqueeze(0).to(DEVICE) for w in eeg_windows_np]
            
            with torch.no_grad():
                logits, ca_history, _, window_logits = self.model(eeg_windows, return_intermediates=True)
            
            self.visualize_constraint_satisfaction(window_logits, logits)
            self.visualize_calcium_evolution(ca_history)
            self.visualize_eeg_heatmaps(eeg_windows_np)
        except Exception as e: messagebox.showerror("Error", f"Failed to process sample:\n{e}"); print(traceback.format_exc())

    def display_image(self, image, coco_id):
        ratio = min(400/image.width, 400/image.height)
        resized = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)
        self.pil_image_tk = ImageTk.PhotoImage(resized)
        self.image_canvas.create_image(200, 200, image=self.pil_image_tk)
        self.coco_id_label.config(text=f"COCO ID: {coco_id}")

    def visualize_constraint_satisfaction(self, window_logits, final_logits):
        self.constraint_fig.clear()
        cat_list = list(TARGET_CATEGORIES.keys())
        n_windows = len(window_logits)
        final_probs = torch.sigmoid(final_logits).squeeze(0).cpu().numpy()
        top_indices = np.argsort(final_probs)[::-1][:10]
        axes = self.constraint_fig.subplots(1, n_windows + 1)
        
        for i, (ax, wl) in enumerate(zip(axes[:-1], window_logits)):
            probs = torch.sigmoid(wl).squeeze(0).cpu().numpy()[top_indices]
            ax.barh([cat_list[idx] for idx in top_indices], probs, color='steelblue')
            ax.set_title(f"{TIME_WINDOWS[i][2]}\n({TIME_WINDOWS[i][0]}-{TIME_WINDOWS[i][1]}ms)", fontsize=10)
            ax.set_xlim(0, 1); ax.invert_yaxis(); ax.tick_params(axis='y', labelsize=8)
        
        axes[-1].barh([cat_list[idx] for idx in top_indices], final_probs[top_indices], color='darkgreen')
        axes[-1].set_title("Final\n(Combined)", fontsize=10); axes[-1].set_xlim(0, 1); axes[-1].invert_yaxis(); axes[-1].tick_params(axis='y', labelsize=8)
        self.constraint_fig.suptitle("Constraint Satisfaction: Predictions Crystallizing Over Time", fontsize=14, fontweight='bold')
        self.constraint_fig.tight_layout(); self.constraint_canvas.draw()

    def visualize_calcium_evolution(self, ca_history):
        self.calcium_fig.clear()
        n_windows = len(ca_history)
        axes = self.calcium_fig.subplots(2, n_windows)
        
        for i, ca_state in enumerate(ca_history):
            ca_np = ca_state.squeeze(0).cpu().numpy()
            top_20_idx = np.argsort(ca_np)[::-1][:20]
            axes[0, i].plot(ca_np, 'r'); axes[0, i].fill_between(range(len(ca_np)), ca_np, color='r', alpha=0.3)
            axes[0, i].set_title(f"{TIME_WINDOWS[i][2]}\n({TIME_WINDOWS[i][0]}-{TIME_WINDOWS[i][1]}ms)", fontsize=10)
            axes[1, i].barh([f"F{idx}" for idx in top_20_idx], ca_np[top_20_idx], color='darkred')
            axes[1, i].invert_yaxis(); axes[1, i].tick_params(axis='y', labelsize=7)
        self.calcium_fig.suptitle("Calcium Attention: What the Model Focuses On", fontsize=14, fontweight='bold')
        self.calcium_fig.tight_layout(); self.calcium_canvas.draw()
    
    def visualize_eeg_heatmaps(self, eeg_windows_np):
        self.eeg_fig.clear()
        n_windows = len(eeg_windows_np)
        axes = self.eeg_fig.subplots(1, n_windows)
        
        for i, (ax, eeg_data) in enumerate(zip(axes, eeg_windows_np)):
            im = ax.imshow(eeg_data, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
            ax.set_title(f"{TIME_WINDOWS[i][2]}\n({TIME_WINDOWS[i][0]}-{TIME_WINDOWS[i][1]}ms)", fontsize=10)
            if i == 0: ax.set_ylabel("Channel")
            self.eeg_fig.colorbar(im, ax=ax) # CORRECTED
        self.eeg_fig.suptitle("Raw EEG Signals by Time Window", fontsize=14, fontweight='bold')
        self.eeg_fig.tight_layout(); self.eeg_canvas.draw()

if __name__ == "__main__":
    app = CalciumBridgeViewer()
    app.mainloop()