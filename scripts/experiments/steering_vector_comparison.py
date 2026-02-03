#!/usr/bin/env python3
"""
Steering Vector Comparison for Temporal Scope

Compares different steering vector methods from AxBench paper (Wu et al., 2025):
- SAE: Sparse Autoencoder features (GemmaScope)
- DiffMean: Difference-in-means between positive/negative classes
- PCA: First principal component of positive representations
- Linear Probe: Learned direction via logistic regression

Applied to temporal scope classification (immediate vs long-term).

Usage:
    python steering_vector_comparison.py
    python steering_vector_comparison.py --layers 6 13 20
    python steering_vector_comparison.py --methods DiffMean PCA Probe
    python steering_vector_comparison.py --ablation-k 16 32 64 128
"""

# Fix for local 'datasets' folder shadowing HuggingFace datasets package
import sys
from pathlib import Path
_src_path = str(Path(__file__).parent.parent.parent / "src")
if _src_path in sys.path:
    sys.path.remove(_src_path)

import torch
import numpy as np
import json
import re
import random
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from transformer_lens import HookedTransformer
from sae_lens import SAE

DEVICE = "cpu"  # GPU requires significant memory
print(f"Using device: {DEVICE}")


# =============================================================================
# CONFIGURATION (for ablation studies)
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for steering vector comparison experiment."""
    # Model settings
    model_name: str = "gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    device: str = "cpu"

    # Layer settings
    layers: List[int] = field(default_factory=lambda: [6, 13, 20, 24])

    # SAE settings
    top_k_latents: int = 64
    sae_width: str = "16k"

    # Methods to compare (subset of: DiffMean, PCA, Probe, LAT, SSV, ReFT-r1, SAE)
    methods: List[str] = field(default_factory=lambda: ["DiffMean", "PCA", "Probe", "LAT", "SSV", "ReFT-r1", "SAE"])

    # Steering demo settings
    demo_prompts: List[str] = field(default_factory=lambda: [
        "I think I'll go to the gym",
        "I'll go take a nap",
        "I'll fill up the gas",
        "I want to pay",
        "Should we tell them the news",
        "I'll do that",
    ])
    steering_alphas: List[float] = field(default_factory=lambda: [0, 10, 50, 100])

    # Data paths (relative to script location)
    train_dataset: str = "temporal_scope_pairs_minimal.json"
    test_dataset: str = "temporal_scope_clean.json"

    # Output settings
    output_dir: str = "steering_vector_comparison"
    save_vectors: bool = True
    verbose: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_temporal_dataset(dataset_path: str, verbose: bool = True) -> Tuple[List[str], np.ndarray, dict]:
    """Load temporal scope pairs dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data['pairs']
    metadata = data.get('metadata', {})

    prompts = []
    labels = []

    for pair in pairs:
        prompts.append(pair['immediate'])
        labels.append(0)  # 0 = immediate
        prompts.append(pair['long_term'])
        labels.append(1)  # 1 = long_term

    if verbose:
        print(f"  Loaded {Path(dataset_path).name}: {len(pairs)} pairs → {len(prompts)} samples")
        print(f"  {prompts[0]} {labels[0]} {prompts[1]} {labels[1]}")

    return prompts, np.array(labels), metadata


def load_temporal_clean_dataset(dataset_path: str, verbose: bool = True) -> Tuple[List[str], np.ndarray, dict]:
    """Load temporal_scope_clean.json - uses both immediate and long_term from each pair."""
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data['pairs']
    metadata = data.get('metadata', {})

    prompts = []
    labels = []

    for pair in pairs:
        question = pair['question']

        # Add immediate version
        immediate_answer = re.sub(r'^\s*\([AB]\)\s*', '', pair['immediate'])
        prompts.append(question + " " + immediate_answer)
        labels.append(0)

        # Add long_term version
        longterm_answer = re.sub(r'^\s*\([AB]\)\s*', '', pair['long_term'])
        prompts.append(question + " " + longterm_answer)
        labels.append(1)

    if verbose:
        print(f"  Loaded {Path(dataset_path).name}: {len(pairs)} pairs → {len(prompts)} samples")
        print(f"  {prompts[0]} {labels[0]} {prompts[1]} {labels[1]}")

    return prompts, np.array(labels), metadata


def load_combined_dataset(data_dir: Path, verbose: bool = True) -> Tuple[List[str], np.ndarray]:
    """Load and combine both temporal datasets for training."""
    if verbose:
        print(f"Loading datasets...")

    all_prompts = []
    all_labels = []

    # Load minimal pairs dataset
    minimal_path = data_dir / "temporal_scope_pairs_minimal.json"
    if minimal_path.exists():
        prompts1, labels1, _ = load_temporal_dataset(str(minimal_path), verbose=verbose)
        all_prompts.extend(prompts1)
        all_labels.extend(labels1)

    # Load clean dataset
    clean_path = data_dir / "temporal_scope_clean.json"
    if clean_path.exists():
        prompts2, labels2, _ = load_temporal_clean_dataset(str(clean_path), verbose=verbose)
        all_prompts.extend(prompts2)
        all_labels.extend(labels2)

    if verbose:
        n_imm = sum(l == 0 for l in all_labels)
        n_long = sum(l == 1 for l in all_labels)
        print(f"  Total: {len(all_prompts)} samples ({n_imm} immediate, {n_long} long-term)")

    return all_prompts, np.array(all_labels)


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

def extract_activations(model: HookedTransformer, prompts: List[str], layer: int,
                        sae: Optional[SAE] = None, batch_size: int = 16,
                        verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract residual stream activations at the last token position.

    Args:
        model: HookedTransformer model
        prompts: List of input prompts
        layer: Layer to extract activations from
        sae: Optional SAE to encode activations through. If provided, also returns SAE latents.
        batch_size: Batch size for processing
        verbose: Whether to print progress

    Returns:
        raw_activations: (n_prompts, d_model) array of activations
        sae_latents: (n_prompts, d_sae) array of SAE latents, or None if sae not provided
    """
    if verbose:
        print(f"Extracting activations (layer {layer}{' + SAE' if sae else ''})...")

    hook_name = f"blocks.{layer}.hook_resid_post"
    all_raw_acts = []
    all_sae_latents = [] if sae else None

    for i in tqdm(range(0, len(prompts), batch_size), desc="  Batches", disable=not verbose):
        batch_prompts = prompts[i:i + batch_size]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_prompts,
                names_filter=[hook_name],
                stop_at_layer=layer + 1,
            )

        acts = cache[hook_name]
        last_token_acts = acts[:, -1, :]
        all_raw_acts.append(last_token_acts.detach().float().cpu().numpy())

        if sae is not None:
            sae_out = sae.encode(last_token_acts)
            all_sae_latents.append(sae_out.detach().float().cpu().numpy())

    raw_activations = np.concatenate(all_raw_acts, axis=0)
    sae_latents = np.concatenate(all_sae_latents, axis=0) if sae else None

    if verbose:
        sae_info = f", SAE sparsity: {(sae_latents != 0).mean():.4f}" if sae_latents is not None else ""
        print(f"  Activations: {raw_activations.shape}{sae_info}")

    return raw_activations, sae_latents


# =============================================================================
# STEERING VECTOR METHODS (from AxBench paper)
# =============================================================================

@dataclass
class SteeringVectorResult:
    """Result from computing a steering vector."""
    name: str
    steering_vector: np.ndarray  # (d_model,)
    detection_scores: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


class SteeringMethod(ABC):
    """Abstract base class for steering vector methods."""

    def __init__(self):
        self.steering_vector = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        """Compute the steering vector from activations and labels."""
        pass

    @abstractmethod
    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        """Compute detection scores for given activations."""
        pass

    def steer(self, activation: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Apply steering to an activation via activation addition.

        Sign convention:
        - steering_vector = mean(long_term) - mean(immediate), pointing toward long_term
        - We ADD the vector scaled by alpha to steer toward long_term
        - Positive alpha → more long_term, negative alpha → more immediate
        """
        if self.steering_vector is None:
            raise ValueError("Must call fit() before steer()")
        return activation + alpha * self.steering_vector

    def get_steering_vector(self) -> np.ndarray:
        """Return the steering vector (d_model,)."""
        if self.steering_vector is None:
            raise ValueError("Must call fit() first")
        return self.steering_vector


class DiffMeanSteering(SteeringMethod):
    """
    Difference-in-means steering vector (AxBench Eq. 4).

    w_DiffMean = mean(H+) - mean(H-)

    Verified
    """

    def __init__(self):
        super().__init__()
        self._name = "DiffMean"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        # Marshall: Verified
        H_pos = activations[labels == 1]
        H_neg = activations[labels == 0]

        mean_pos = H_pos.mean(axis=0)
        mean_neg = H_neg.mean(axis=0)

        print(f'{mean_pos.shape=}, {mean_neg.shape=}')

        self.steering_vector = mean_pos - mean_neg

        norm = np.linalg.norm(self.steering_vector)
        if norm > 0:
            self.steering_vector = self.steering_vector / norm
        else:
            raise ValueError("Normalization failed")

        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={'n_positive': len(H_pos), 'n_negative': len(H_neg), 'norm_before_normalize': norm}
        )

    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        if self.steering_vector is None:
            raise ValueError("Must call fit() before detect()")
        return activations @ self.steering_vector


class PCASteering(SteeringMethod):
    """
    PCA-based steering vector.

    Uses first principal component of positive representations.

    verified
    use_all_data versus just use positive set. See footnote 8 in Axbench paper where they said no difference was found
    """

    def __init__(self, use_all_data: bool = False):
        super().__init__()
        self.pca = None
        self.use_all_data = use_all_data
        self._name = "PCA"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        if self.use_all_data:
            H = activations
        else:
            H = activations[labels == 1]

        mean = H.mean(axis=0)
        H_centered = H - mean

        print(f'{H.shape=} {mean.shape=} {H_centered.shape=}')

        self.pca = PCA(n_components=1, random_state=42)
        self.pca.fit(H_centered)

        self.steering_vector = self.pca.components_[0]

        # Ensure consistent direction; apparently PCA is direction invariant? Makes sense
        pos_proj = (activations[labels == 1] @ self.steering_vector).mean()
        neg_proj = (activations[labels == 0] @ self.steering_vector).mean()
        if pos_proj < neg_proj:
            self.steering_vector = -self.steering_vector

        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={'explained_variance_ratio': self.pca.explained_variance_ratio_[0], 'use_all_data': self.use_all_data}
        )

    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        if self.steering_vector is None:
            raise ValueError("Must call fit() before detect()")
        return activations @ self.steering_vector


class LinearProbeSteering(SteeringMethod):
    """
    Linear Probe steering vector (AxBench Eq. 5).

    Learns direction via logistic regression.
    """

    def __init__(self, max_iter: int = 5000, C: float = 1.0, use_scaling: bool = True):
        super().__init__()
        self.probe = None
        self.scaler = None
        self.max_iter = max_iter
        self.C = C
        self.use_scaling = use_scaling
        self._name = "Probe"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        if self.use_scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(activations)
        else:
            X = activations

        self.probe = LogisticRegression(max_iter=self.max_iter, C=self.C, random_state=42, solver='lbfgs')
        self.probe.fit(X, labels)

        # TODO: double check this
        w = self.probe.coef_[0]
        if self.use_scaling:
            self.steering_vector = w / self.scaler.scale_
        else:
            self.steering_vector = w

        norm = np.linalg.norm(self.steering_vector)
        if norm > 0:
            self.steering_vector = self.steering_vector / norm
        else:
            raise ValueError("Normalization failed")

        train_acc = self.probe.score(X, labels)

        # TODO: Judging from results, it feels linear probe is getting the wrong sign
        return SteeringVectorResult(
            name=self.name,
            steering_vector=-self.steering_vector,
            metadata={'train_accuracy': train_acc, 'C': self.C, 'use_scaling': self.use_scaling}
        )

    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        if self.steering_vector is None:
            raise ValueError("Must call fit() before detect()")
        return activations @ self.steering_vector


class LATSteering(SteeringMethod):
    """
    Linear Artificial Tomography (LAT) steering vector.

    From AxBench paper (Zou et al., 2023):
    - Creates pairwise activation differences from positive examples
    - Normalizes each difference to unit length
    - Uses first principal component of these differences as steering vector
    """

    def __init__(self):
        super().__init__()
        self.pca = None
        self._name = "LAT"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        # Use only positive examples (following AxBench)
        H_pos = activations[labels == 1]
        n = len(H_pos)

        if n < 2:
            raise ValueError("Need at least 2 positive examples for LAT")

        # Create pairwise differences
        # Randomly partition into pairs
        indices = np.random.permutation(n)
        n_pairs = n // 2
        diffs = []

        for i in range(n_pairs):
            h_i = H_pos[indices[2 * i]]
            h_j = H_pos[indices[2 * i + 1]]
            delta = h_i - h_j
            norm = np.linalg.norm(delta)
            if norm > 1e-8:
                diffs.append(delta / norm)  # Unit normalize each difference

        if len(diffs) < 2:
            raise ValueError("Not enough valid pairs for LAT")

        diffs_matrix = np.stack(diffs)  # (n_pairs, d_model)

        # PCA on the differences
        self.pca = PCA(n_components=1, random_state=42)
        self.pca.fit(diffs_matrix)

        self.steering_vector = self.pca.components_[0]

        # Ensure consistent direction (positive examples should project positively)
        pos_proj = (activations[labels == 1] @ self.steering_vector).mean()
        neg_proj = (activations[labels == 0] @ self.steering_vector).mean()
        if pos_proj < neg_proj:
            self.steering_vector = -self.steering_vector

        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={
                'n_pairs': len(diffs),
                'explained_variance_ratio': float(self.pca.explained_variance_ratio_[0])
            }
        )

    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        if self.steering_vector is None:
            raise ValueError("Must call fit() before detect()")
        return activations @ self.steering_vector


class SSVSteering(SteeringMethod):
    """
    Supervised Steering Vector (SSV).

    From AxBench paper (Eq. 6-7):
    - Intervention: Φ^SSV(hi) = hi + wSSV
    - Training: minimize LM loss on positive examples with intervention applied
    - min_{wSSV} { Σ log P_LM(y_t | y_{<t}, x; h ← Φ^SSV(h)) }

    Requires backprop through the LM to learn wSSV.
    """

    def __init__(self, n_iterations: int = 20, lr: float = 0.5, batch_size: int = 4):
        super().__init__()
        self.n_iterations = n_iterations
        self.lr = lr
        self.batch_size = batch_size
        self._name = "SSV"
        self.requires_lm = True  # Flag to indicate this method needs LM access

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        """Fallback fit using activations only (not recommended - use fit_with_lm)."""
        # Initialize with DiffMean as fallback
        H_pos = activations[labels == 1]
        H_neg = activations[labels == 0]
        w = H_pos.mean(axis=0) - H_neg.mean(axis=0)
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        self.steering_vector = w
        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={'fallback': True, 'note': 'Used DiffMean fallback - fit_with_lm not called'}
        )

    def fit_with_lm(self, model: "HookedTransformer", prompts: List[str], labels: np.ndarray,
                   layer: int, device: str = "cpu") -> SteeringVectorResult:
        """
        Train SSV by backpropagating LM loss through the intervention.

        Only trains on positive examples (labels == 1).
        """
        print(f"  Training SSV with LM backprop (layer {layer}, {self.n_iterations} iterations)...")

        # Get positive prompts only (SSV trains on positive examples)
        pos_indices = np.where(labels == 1)[0]
        pos_prompts = [prompts[i] for i in pos_indices]

        d_model = model.cfg.d_model
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Initialize w with small random values
        w = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)

        # Use DiffMean for initialization (better starting point)
        with torch.no_grad():
            pos_acts = []
            neg_acts = []
            for i, prompt in enumerate(prompts[:min(50, len(prompts))]):  # Sample for init
                _, cache = model.run_with_cache(prompt, names_filter=[hook_name], stop_at_layer=layer+1)
                act = cache[hook_name][0, -1, :].float()
                if labels[i] == 1:
                    pos_acts.append(act)
                else:
                    neg_acts.append(act)
            if pos_acts and neg_acts:
                init_w = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
                init_w = init_w / (init_w.norm() + 1e-8)
                w.data.copy_(init_w * 0.1)  # Scale down for stability

        optimizer = torch.optim.Adam([w], lr=self.lr)

        losses = []
        for iteration in range(self.n_iterations):
            # Sample a batch of positive prompts
            batch_indices = np.random.choice(len(pos_prompts), min(self.batch_size, len(pos_prompts)), replace=False)
            batch_prompts = [pos_prompts[i] for i in batch_indices]

            total_loss = 0.0
            optimizer.zero_grad()

            for prompt in batch_prompts:
                # Tokenize
                tokens = model.to_tokens(prompt)
                if tokens.shape[1] < 2:
                    continue

                # Define steering hook
                def steering_hook(activations, hook, steering_vec=w):
                    # Add steering vector to all positions
                    return activations + steering_vec.unsqueeze(0).unsqueeze(0)

                # Forward pass with intervention
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, steering_hook)]
                )

                # Compute LM loss (predict next token)
                # Shift logits and tokens for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tokens[:, 1:].contiguous()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                total_loss = total_loss + loss

            if total_loss > 0:
                avg_loss = total_loss / len(batch_prompts)
                avg_loss.backward()
                optimizer.step()
                losses.append(avg_loss.item())

            if (iteration + 1) % 10 == 0:
                print(f"    Iteration {iteration+1}/{self.n_iterations}, Loss: {losses[-1]:.4f}")

        # Normalize final vector
        with torch.no_grad():
            w_final = w.detach().cpu().numpy()
            norm = np.linalg.norm(w_final)
            if norm > 0:
                w_final = w_final / norm

        self.steering_vector = w_final

        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={
                'n_iterations': self.n_iterations,
                'lr': self.lr,
                'final_loss': losses[-1] if losses else None,
                'n_positive_samples': len(pos_prompts)
            }
        )

    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        if self.steering_vector is None:
            raise ValueError("Must call fit() before detect()")
        # SSV uses ReLU for detection scores
        scores = activations @ self.steering_vector
        return np.maximum(scores, 0)  # ReLU


class ReFTr1Steering(SteeringMethod):
    """
    Rank-1 Representation Finetuning (ReFT-r1).

    From AxBench paper (Eq. 8-10):
    - Detection: Ψ(hi) = ReLU(hi · w)
    - Intervention: Φ(hi) = hi + (1/k * ||TopK(Ψ(h))||_1) * w
    - Training: min_{w} { -Σ log P_LM(y_t | y_{<t}, x) + λ Σ_{a_i ∉ TopK} ||a_i||_1 }

    Jointly learns detection and steering via LM backprop.
    """

    def __init__(self, top_k: int = 32, n_iterations: int = 20, lr: float = 0.5,
                 lambda_l1: float = 0.01, batch_size: int = 4):
        super().__init__()
        self.top_k = top_k
        self.n_iterations = n_iterations
        self.lr = lr
        self.lambda_l1 = lambda_l1
        self.batch_size = batch_size
        self._name = "ReFT-r1"
        self.requires_lm = True

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray, **kwargs) -> SteeringVectorResult:
        """Fallback fit using activations only (not recommended - use fit_with_lm)."""
        H_pos = activations[labels == 1]
        H_neg = activations[labels == 0]
        w = H_pos.mean(axis=0) - H_neg.mean(axis=0)
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        self.steering_vector = w
        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={'fallback': True, 'note': 'Used DiffMean fallback - fit_with_lm not called'}
        )

    def fit_with_lm(self, model: "HookedTransformer", prompts: List[str], labels: np.ndarray,
                   layer: int, device: str = "cpu") -> SteeringVectorResult:
        """
        Train ReFT-r1 by backpropagating LM loss + L1 regularization.

        Uses positive examples for LM loss, applies TopK-based intervention scaling.
        """
        print(f"  Training ReFT-r1 with LM backprop (layer {layer}, {self.n_iterations} iterations)...")

        pos_indices = np.where(labels == 1)[0]
        pos_prompts = [prompts[i] for i in pos_indices]

        d_model = model.cfg.d_model
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Initialize w
        w = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)

        # Initialize with DiffMean
        with torch.no_grad():
            pos_acts = []
            neg_acts = []
            for i, prompt in enumerate(prompts[:min(50, len(prompts))]):
                _, cache = model.run_with_cache(prompt, names_filter=[hook_name], stop_at_layer=layer+1)
                act = cache[hook_name][0, -1, :].float()
                if labels[i] == 1:
                    pos_acts.append(act)
                else:
                    neg_acts.append(act)
            if pos_acts and neg_acts:
                init_w = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
                init_w = init_w / (init_w.norm() + 1e-8)
                w.data.copy_(init_w * 0.1)

        optimizer = torch.optim.Adam([w], lr=self.lr)

        losses = []
        for iteration in range(self.n_iterations):
            batch_indices = np.random.choice(len(pos_prompts), min(self.batch_size, len(pos_prompts)), replace=False)
            batch_prompts = [pos_prompts[i] for i in batch_indices]

            total_loss = 0.0
            total_l1 = 0.0
            optimizer.zero_grad()

            for prompt in batch_prompts:
                tokens = model.to_tokens(prompt)
                if tokens.shape[1] < 2:
                    continue

                # Store detection scores for L1 regularization
                detection_scores_list = []

                def reft_hook(activations, hook, steering_vec=w, scores_list=detection_scores_list):
                    # Compute detection scores: Ψ(hi) = ReLU(hi · w)
                    # activations: (batch, seq, d_model)
                    scores = torch.relu(torch.einsum('bsd,d->bs', activations.float(), steering_vec))
                    scores_list.append(scores)

                    # Compute intervention scaling: (1/k * ||TopK(Ψ(h))||_1)
                    flat_scores = scores.flatten()
                    if len(flat_scores) > self.top_k:
                        topk_vals, _ = torch.topk(flat_scores, self.top_k)
                        scale = topk_vals.sum() / self.top_k
                    else:
                        scale = flat_scores.sum() / (len(flat_scores) + 1e-8)

                    # Intervention: Φ(hi) = hi + scale * w
                    return activations + scale * steering_vec.unsqueeze(0).unsqueeze(0)

                # Forward pass with ReFT intervention
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, reft_hook)]
                )

                # LM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tokens[:, 1:].contiguous()
                lm_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )

                # L1 regularization on non-top-k detection scores
                if detection_scores_list:
                    all_scores = torch.cat([s.flatten() for s in detection_scores_list])
                    if len(all_scores) > self.top_k:
                        _, topk_indices = torch.topk(all_scores, self.top_k)
                        mask = torch.ones_like(all_scores, dtype=torch.bool)
                        mask[topk_indices] = False
                        non_topk_scores = all_scores[mask]
                        l1_loss = self.lambda_l1 * non_topk_scores.abs().mean()
                    else:
                        l1_loss = 0.0
                else:
                    l1_loss = 0.0

                loss = lm_loss + l1_loss
                total_loss = total_loss + lm_loss
                total_l1 = total_l1 + (l1_loss if isinstance(l1_loss, float) else l1_loss.item())

            if total_loss > 0:
                avg_loss = total_loss / len(batch_prompts)
                avg_loss.backward()
                optimizer.step()
                losses.append(avg_loss.item())

            if (iteration + 1) % 10 == 0:
                print(f"    Iteration {iteration+1}/{self.n_iterations}, LM Loss: {losses[-1]:.4f}")

        # Normalize final vector
        with torch.no_grad():
            w_final = w.detach().cpu().numpy()
            norm = np.linalg.norm(w_final)
            if norm > 0:
                w_final = w_final / norm

        self.steering_vector = w_final

        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={
                'top_k': self.top_k,
                'n_iterations': self.n_iterations,
                'lr': self.lr,
                'lambda_l1': self.lambda_l1,
                'final_loss': losses[-1] if losses else None
            }
        )

    def detect(self, activations: np.ndarray, **kwargs) -> np.ndarray:
        if self.steering_vector is None:
            raise ValueError("Must call fit() before detect()")
        # ReFT-r1 uses ReLU for detection
        return np.maximum(activations @ self.steering_vector, 0)


class SAESteering(SteeringMethod):
    """
    SAE-based steering using top-k discriminative latents.

    Uses top-k SAE latents with highest class difference,
    then computes steering vector from SAE decoder directions.

    Note: This differs from AxBench's single-feature SAE approach. AxBench uses
    one SAE latent per concept (matched by Neuronpedia label or AUROC selection).
    This implementation uses a weighted combination of top-k discriminative features,
    which is better suited for abstract concepts like "temporal scope" that may not
    map to a single SAE feature.
    """

    def __init__(self, sae: SAE, k: int = 64):
        super().__init__()
        self.sae = sae
        self.k = k
        self.top_k_indices = None
        self.signed_diff = None
        self._name = "SAE"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, activations: np.ndarray, labels: np.ndarray,
            sae_latents: np.ndarray = None, **kwargs) -> SteeringVectorResult:
        if sae_latents is None:
            with torch.no_grad():
                acts_tensor = torch.tensor(activations, dtype=torch.float32, device=self.sae.device)
                sae_latents = self.sae.encode(acts_tensor).cpu().numpy()

        mean_T0 = sae_latents[labels == 0].mean(axis=0)
        mean_T1 = sae_latents[labels == 1].mean(axis=0)

        self.signed_diff = mean_T1 - mean_T0
        abs_diff = np.abs(self.signed_diff)

        self.top_k_indices = np.argsort(abs_diff)[-self.k:][::-1]

        W_dec = self.sae.W_dec.detach().cpu().numpy()
        steering_directions = W_dec[self.top_k_indices]
        weights = self.signed_diff[self.top_k_indices]

        self.steering_vector = (weights[:, None] * steering_directions).sum(axis=0)
        print(f' SAE {self.steering_vector.shape=}')
        norm = np.linalg.norm(self.steering_vector)
        if norm > 0:
            self.steering_vector = self.steering_vector / norm
        else:
            raise ValueError("Normalization failed")

        return SteeringVectorResult(
            name=self.name,
            steering_vector=self.steering_vector,
            metadata={'k': self.k, 'top_k_indices': self.top_k_indices.tolist()[:10], 'sparsity': (sae_latents != 0).mean()}
        )

    def detect(self, activations: np.ndarray, sae_latents: np.ndarray = None, **kwargs) -> np.ndarray:
        if self.top_k_indices is None:
            raise ValueError("Must call fit() before detect()")

        if sae_latents is None:
            with torch.no_grad():
                acts_tensor = torch.tensor(activations, dtype=torch.float32, device=self.sae.device)
                sae_latents = self.sae.encode(acts_tensor).cpu().numpy()

        selected = sae_latents[:, self.top_k_indices]
        weights = self.signed_diff[self.top_k_indices]
        return (selected * weights).sum(axis=1)


def create_method(name: str, sae: SAE = None, **kwargs) -> SteeringMethod:
    """Factory function to create steering methods."""
    if name == "DiffMean":
        return DiffMeanSteering()
    elif name == "PCA":
        return PCASteering(use_all_data=kwargs.get('pca_use_all_data', False))
    elif name == "Probe":
        return LinearProbeSteering(
            max_iter=kwargs.get('probe_max_iter', 5000),
            C=kwargs.get('probe_C', 1.0),
            use_scaling=kwargs.get('probe_use_scaling', True)
        )
    elif name == "LAT":
        return LATSteering()
    elif name == "SSV":
        return SSVSteering(
            n_iterations=kwargs.get('ssv_iterations', 100),
            lr=kwargs.get('ssv_lr', 0.01)
        )
    elif name == "ReFT-r1":
        return ReFTr1Steering(
            top_k=kwargs.get('reft_top_k', 32),
            n_iterations=kwargs.get('reft_iterations', 200),
            lr=kwargs.get('reft_lr', 0.01),
            lambda_l1=kwargs.get('reft_lambda_l1', 0.01)
        )
    elif name == "SAE":
        if sae is None:
            raise ValueError("SAE method requires sae argument")
        return SAESteering(sae, k=kwargs.get('top_k_latents', 64))
    else:
        raise ValueError(f"Unknown method: {name}")


# =============================================================================
# STEERING VECTOR ANALYSIS
# =============================================================================

def compute_vector_similarities(vectors: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarities between steering vectors.

    Args:
        vectors: Dict mapping method name to steering vector (d_model,)

    Returns:
        similarity_matrix: (n_methods, n_methods) cosine similarity matrix
    """
    names = list(vectors.keys())
    n = len(names)
    similarity_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            v_i = vectors[name_i]
            v_j = vectors[name_j]

            # Cosine similarity
            dot = np.dot(v_i, v_j)
            norm_i = np.linalg.norm(v_i)
            norm_j = np.linalg.norm(v_j)

            if norm_i > 0 and norm_j > 0:
                similarity_matrix[i, j] = dot / (norm_i * norm_j)
            else:
                similarity_matrix[i, j] = 0.0

    return similarity_matrix, names


def analyze_steering_vectors(vectors: Dict[str, np.ndarray], output_dir: Path,
                              layer: int, verbose: bool = True):
    """
    Analyze and visualize steering vectors.

    1. Cosine similarity matrix
    2. PCA projection of vectors
    """
    names = list(vectors.keys())
    n_methods = len(names)
    d_model = vectors[names[0]].shape[0]

    # 1. Cosine similarity matrix
    sim_matrix, _ = compute_vector_similarities(vectors)

    if verbose:
        print(f"\nCosine similarities: ", end="")
        pairs = []
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                pairs.append(f"{names[i]}↔{names[j]}={sim_matrix[i,j]:.2f}")
        print(", ".join(pairs))

    # 2. PCA projection
    vectors_matrix = np.stack([vectors[name] for name in names])  # (n_methods, d_model)
    pca = PCA(n_components=min(2, n_methods))
    vectors_2d = pca.fit_transform(vectors_matrix)

    # Plot similarity matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_methods))
    ax.set_yticks(range(n_methods))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_title(f'Cosine Similarity (Layer {layer})')

    for i in range(n_methods):
        for j in range(n_methods):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                   color='white' if abs(sim_matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax)

    # PCA scatter
    ax = axes[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_methods))
    for i, name in enumerate(names):
        if vectors_2d.shape[1] >= 2:
            ax.scatter(vectors_2d[i, 0], vectors_2d[i, 1], c=[colors[i]], s=200, label=name)
            ax.annotate(name, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10,
                       xytext=(5, 5), textcoords='offset points')
        else:
            ax.scatter(vectors_2d[i, 0], 0, c=[colors[i]], s=200, label=name)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    if vectors_2d.shape[1] >= 2:
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA of Steering Vectors (Layer {layer})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / f"steering_vectors_analysis_layer{layer}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved vector analysis plot to {plot_path}")
    plt.close()

    return {
        'similarity_matrix': sim_matrix.tolist(),
        'method_names': names,
        'pca_coords': vectors_2d.tolist(),
        'pca_explained_variance': pca.explained_variance_ratio_.tolist()
    }

# =============================================================================
# STEERING DEMO (matching sae_temporal_probing.py format)
# =============================================================================

def run_steering_demo(model: HookedTransformer, methods: Dict[str, SteeringMethod],
                       layer: int, config: ExperimentConfig):
    """
    Run steering demonstration comparing P(' now') vs P(' later').

    Matches the format from sae_temporal_probing.py.
    """
    print(f"\nSteering demo (layer {layer}): P(' now') vs P(' later')")

    # Get token IDs for temporal words (note the leading space)
    temporal_tokens = {
        " now": model.tokenizer.encode(" now", add_special_tokens=False)[0],
        " later": model.tokenizer.encode(" later", add_special_tokens=False)[0],
        # " soon": model.tokenizer.encode(" soon", add_special_tokens=False)[0],
        # " today": model.tokenizer.encode(" today", add_special_tokens=False)[0],
        # " tomorrow": model.tokenizer.encode(" tomorrow", add_special_tokens=False)[0],
        # " eventually": model.tokenizer.encode(" eventually", add_special_tokens=False)[0],
    }

    hook_name = f"blocks.{layer}.hook_resid_post"
    alphas = config.steering_alphas

    # Store results for plotting
    all_results = []

    # Use both positive alphas (toward long_term) and negative (toward immediate)
    # Alpha > 0 steers toward long_term, alpha < 0 steers toward immediate
    test_alphas = []
    for a in alphas:
        if a != 0:
            test_alphas.extend([a, -a])  # positive and negative
    test_alphas = sorted(set(test_alphas))

    for demo_prompt in config.demo_prompts:
        print(f"\n  \"{demo_prompt}\"")
        print(f"  {'Method':<8} {'Alpha':>6} {'P(now)':>10} {'P(later)':>10} {'Ratio':>8}")

        prompt_results = {'prompt': demo_prompt, 'results': []}

        # Unsteered baseline
        with torch.no_grad():
            logits = model(demo_prompt)
        probs = torch.softmax(logits[0, -1], dim=-1)

        p_now = probs[temporal_tokens[" now"]].item()
        p_later = probs[temporal_tokens[" later"]].item()
        ratio = p_now / p_later if p_later > 0 else float('inf')

        print(f"  {'Baseline':<8} {0:>6} {p_now:>10.6f} {p_later:>10.6f} {ratio:>8.2f}")
        prompt_results['results'].append({
            'method': 'Baseline', 'alpha': 0,
            'p_now': p_now, 'p_later': p_later, 'ratio': ratio
        })

        # Test each method with positive and negative alphas
        for method_name, method in methods.items():
            for alpha in test_alphas:
                def make_steering_hook(m, a):
                    def hook(activations, hook=None):
                        last_act = activations[:, -1, :].cpu().numpy()
                        steered = m.steer(last_act, alpha=a)
                        activations[:, -1, :] = torch.tensor(
                            steered, dtype=activations.dtype, device=activations.device
                        )
                        return activations
                    return hook

                with torch.no_grad():
                    logits = model.run_with_hooks(
                        demo_prompt,
                        fwd_hooks=[(hook_name, make_steering_hook(method, alpha))]
                    )
                probs = torch.softmax(logits[0, -1], dim=-1)

                p_now = probs[temporal_tokens[" now"]].item()
                p_later = probs[temporal_tokens[" later"]].item()
                ratio = p_now / p_later if p_later > 0 else float('inf')

                print(f"  {method_name:<8} {alpha:>6} {p_now:>10.6f} {p_later:>10.6f} {ratio:>8.2f}")
                prompt_results['results'].append({
                    'method': method_name, 'alpha': alpha,
                    'p_now': p_now, 'p_later': p_later, 'ratio': ratio
                })

        all_results.append(prompt_results)

    return all_results


def plot_steering_demo(demo_results: List[dict], output_dir: Path, layer: int):
    """
    Create visualization of steering demo results.

    Shows P(now)/P(later) ratio vs alpha for each method.
    Positive alpha → toward long_term, negative alpha → toward immediate.
    """
    n_prompts = len(demo_results)
    if n_prompts == 0:
        return

    # Color scheme with distinct colors
    colors = {
        'Baseline': '#7f7f7f',  # gray
        'DiffMean': '#1f77b4',  # blue
        'PCA': '#ff7f0e',       # orange
        'Probe': '#2ca02c',     # green
        'LAT': '#d62728',       # red
        'SSV': '#9467bd',       # purple
        'ReFT-r1': '#8c564b',   # brown
        'SAE': '#e377c2'        # pink
    }

    # Create figure: one row, columns = prompts
    n_cols = n_prompts
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)

    # Track handles/labels for shared legend
    legend_handles = []
    legend_labels = []
    legend_collected = False

    for col_idx, prompt_data in enumerate(demo_results):
        ax = axes[0, col_idx]
        prompt = prompt_data['prompt']
        results = prompt_data['results']

        # Group results by method
        methods_data = {}
        for r in results:
            method = r['method']
            if method not in methods_data:
                methods_data[method] = []
            methods_data[method].append((r['alpha'], r['p_now'], r['p_later']))

        # Plot each method
        for method, data in methods_data.items():
            color = colors.get(method, 'black')

            # Sort by alpha
            data_sorted = sorted(data, key=lambda x: x[0])
            alphas = [d[0] for d in data_sorted]
            ratios = [d[1] / d[2] if d[2] > 0 else 10 for d in data_sorted]

            if method == 'Baseline':
                # Show baseline as horizontal dashed line
                ax.axhline(y=ratios[0], color=color, linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
                if not legend_collected:
                    legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2))
                    legend_labels.append('Baseline')
            else:
                line, = ax.plot(alphas, ratios, 'o-', color=color, linewidth=2, markersize=5)
                if not legend_collected:
                    legend_handles.append(line)
                    legend_labels.append(method)

        legend_collected = True

        # Reference line at ratio = 1
        ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.3, linewidth=1)

        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlabel('Alpha (+ → long-term, − → immediate)', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('P(now) / P(later)', fontsize=10)

        title = f'"{prompt[:25]}..."' if len(prompt) > 25 else f'"{prompt}"'
        ax.set_title(title, fontsize=9)

    # Single shared legend at bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.08))

    plt.suptitle(f'Steering Demo: Layer {layer}', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    plot_path = output_dir / f"steering_demo_layer{layer}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved steering demo plot to {plot_path}")
    plt.close()


def plot_steering_probs(demo_results: List[dict], output_dir: Path, layer: int):
    """
    Create bar chart visualization comparing P(now) and P(later) across conditions.
    """
    # Use first prompt for detailed visualization
    if not demo_results:
        return

    prompt_data = demo_results[0]
    prompt = prompt_data['prompt']
    results = prompt_data['results']

    # Filter to baseline and extreme alphas (±100)
    filtered = [r for r in results if r['alpha'] in [0, 100, -100]]

    n_bars = len(filtered)
    x = np.arange(n_bars)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    p_nows = [r['p_now'] for r in filtered]
    p_laters = [r['p_later'] for r in filtered]
    labels = [f"{r['method']}\nα={r['alpha']}" for r in filtered]

    bars1 = ax.bar(x - width/2, p_nows, width, label="P(' now')", color='orangered')
    bars2 = ax.bar(x + width/2, p_laters, width, label="P(' later')", color='steelblue')

    ax.set_ylabel('Probability')
    ax.set_title(f'Steering Effects on Temporal Token Probabilities\nPrompt: "{prompt}"')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0.001:
            ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    for bar in bars2:
        height = bar.get_height()
        if height > 0.001:
            ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plot_path = output_dir / f"steering_probs_layer{layer}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved probability comparison plot to {plot_path}")
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run the full steering vector comparison experiment.
    Focused on steering demo - no AUROC/accuracy evaluation.
    """
    print(f"Steering Vector Comparison: {', '.join(config.methods)} on layers {config.layers}")

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data" / "raw"
    output_dir = script_dir.parent.parent / "results" / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load combined dataset for training
    train_prompts, train_labels = load_combined_dataset(data_dir, verbose=config.verbose)

    # Load model
    print(f"\nLoading model: {config.model_name}")
    model = HookedTransformer.from_pretrained(config.model_name, device=config.device)
    print(f"  Layers: {model.cfg.n_layers}, Hidden dim: {model.cfg.d_model}")

    all_results = {'config': config.to_dict(), 'layers': {}}

    for layer in config.layers:
        print(f"\n=== Layer {layer} ===")

        layer_results = {'layer': layer, 'methods': {}, 'vectors': {}}

        # Load SAE if needed
        sae = None
        sae_train_latents = None

        if "SAE" in config.methods:
            sae_id = f"layer_{layer}/width_{config.sae_width}/canonical"
            try:
                sae = SAE.from_pretrained(release=config.sae_release, sae_id=sae_id, device=config.device)
            except Exception as e:
                print(f"  Warning: Failed to load SAE: {e}")
                config.methods = [m for m in config.methods if m != "SAE"]

        # Extract activations (with SAE latents if SAE is available)
        train_acts, sae_train_latents = extract_activations(
            model, train_prompts, layer, sae=sae, verbose=config.verbose
        )

        # Create and fit methods
        methods = {}
        steering_vectors = {}

        for method_name in config.methods:

            method = create_method(method_name, sae=sae, top_k_latents=config.top_k_latents)

            # Check if method requires LM access (SSV and ReFT-r1)
            if hasattr(method, 'requires_lm') and method.requires_lm:
                result = method.fit_with_lm(
                    model=model,
                    prompts=train_prompts,
                    labels=train_labels,
                    layer=layer,
                    device=config.device
                )
            elif isinstance(method, SAESteering):
                result = method.fit(train_acts, train_labels, sae_latents=sae_train_latents)
            else:
                result = method.fit(train_acts, train_labels)

            methods[method_name] = method
            steering_vectors[method_name] = method.get_steering_vector()

            # Brief summary
            meta_str = ", ".join(f"{k}={v}" for k, v in list(result.metadata.items())[:3])
            print(f"  {method_name}: {meta_str}")

            layer_results['methods'][method_name] = {
                'metadata': result.metadata
            }

        # Analyze steering vectors (cosine similarity, PCA)
        if len(steering_vectors) > 1:
            vector_analysis = analyze_steering_vectors(
                steering_vectors, output_dir, layer, verbose=config.verbose
            )
            layer_results['vector_analysis'] = vector_analysis

        # Save steering vectors if requested
        if config.save_vectors:
            layer_results['vectors'] = {name: v.tolist() for name, v in steering_vectors.items()}

        all_results['layers'][layer] = layer_results

        # Run steering demo for this layer
        demo_results = run_steering_demo(model, methods, layer, config)
        layer_results['steering_demo'] = demo_results

        # Plot steering demo results
        plot_steering_demo(demo_results, output_dir, layer)
        plot_steering_probs(demo_results, output_dir, layer)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    results_file = output_dir / "steering_comparison_results.json"

    def make_serializable(obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    serializable_results = make_serializable(all_results)
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return all_results


# =============================================================================
# ABLATION STUDY HELPERS
# =============================================================================

def run_ablation_top_k(config: ExperimentConfig, k_values: List[int]) -> dict:
    """Run ablation study varying top-k SAE latents."""
    print(f"\n{'='*70}")
    print("ABLATION STUDY: Top-K SAE Latents")
    print(f"{'='*70}")
    print(f"K values: {k_values}")

    all_results = {}
    for k in k_values:
        print(f"\n--- K = {k} ---")
        ablation_config = ExperimentConfig(
            **{**config.to_dict(), 'top_k_latents': k, 'output_dir': f"{config.output_dir}_k{k}"}
        )
        all_results[k] = run_experiment(ablation_config)

    return all_results


def run_ablation_layers(config: ExperimentConfig, layer_sets: List[List[int]]) -> dict:
    """Run ablation study with different layer configurations."""
    print(f"\n{'='*70}")
    print("ABLATION STUDY: Layers")
    print(f"{'='*70}")

    all_results = {}
    for layers in layer_sets:
        print(f"\n--- Layers = {layers} ---")
        ablation_config = ExperimentConfig(
            **{**config.to_dict(), 'layers': layers}
        )
        all_results[tuple(layers)] = run_experiment(ablation_config)

    return all_results


def run_ablation_methods(config: ExperimentConfig) -> dict:
    """Run ablation comparing individual methods."""
    print(f"\n{'='*70}")
    print("ABLATION STUDY: Individual Methods")
    print(f"{'='*70}")

    all_methods = ["DiffMean", "PCA", "Probe", "SAE"]
    all_results = {}

    for method in all_methods:
        print(f"\n--- Method = {method} ---")
        ablation_config = ExperimentConfig(
            **{**config.to_dict(), 'methods': [method]}
        )
        all_results[method] = run_experiment(ablation_config)

    return all_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Steering Vector Comparison Experiment")

    parser.add_argument('--model', type=str, default="gemma-2-2b", help="Model name")
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 13, 20, 25], help="Layers to probe")
    # parser.add_argument('--methods', type=str, nargs='+', default=["DiffMean", "PCA", "Probe", "LAT", "SSV", "ReFT-r1", "SAE"],
    #                     help="Methods to compare")
    parser.add_argument('--methods', type=str, nargs='+', default=["DiffMean", "PCA", "LAT", "Probe", "SAE"],
                        help="Methods to compare")

    parser.add_argument('--top-k', type=int, default=64, help="Top-k SAE latents")
    parser.add_argument('--alphas', type=float, nargs='+', default=[0, 10, 50, 100], help="Steering alphas")
    parser.add_argument('--output-dir', type=str, default="steering_vector_comparison", help="Output directory")
    parser.add_argument('--no-save-vectors', action='store_true', help="Don't save steering vectors")
    parser.add_argument('--quiet', action='store_true', help="Less verbose output")

    # Ablation study flags
    parser.add_argument('--ablation-k', type=int, nargs='+', help="Run ablation on top-k values")

    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        model_name=args.model,
        layers=args.layers,
        methods=args.methods,
        top_k_latents=args.top_k,
        steering_alphas=args.alphas,
        output_dir=args.output_dir,
        save_vectors=not args.no_save_vectors,
        verbose=not args.quiet
    )

    print(f"Steering Vector Comparison (AxBench methodology)")
    print(f"Methods: {', '.join(config.methods)} | Layers: {config.layers}")

    if args.ablation_k:
        run_ablation_top_k(config, args.ablation_k)
    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
