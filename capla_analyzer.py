"""
CAPLA Threshold Visualization and Analysis Script

This script provides tools to visualize and analyze CAPLA's adaptive thresholds
and their impact on pseudolabel assignment during semi-supervised training.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import torch


class CAPLAAnalyzer:
    """Analyze CAPLA adaptive thresholds and pseudolabel assignment."""
    
    def __init__(self, class_names: Dict[int, str]):
        """
        Initialize CAPLA analyzer.
        
        Args:
            class_names: Dictionary mapping class IDs to class names
        """
        self.class_names = class_names
        self.threshold_history = {i: [] for i in range(len(class_names))}
        self.score_distributions = {i: [] for i in range(len(class_names))}
        self.update_iterations = []
        
    def record_thresholds(self, iteration: int, thresholds: Dict[int, float]):
        """Record threshold values at a specific iteration."""
        self.update_iterations.append(iteration)
        for class_id, threshold in thresholds.items():
            self.threshold_history[class_id].append(threshold)
    
    def record_scores(self, class_id: int, scores: List[float]):
        """Record pseudolabel scores for a class."""
        self.score_distributions[class_id].extend(scores)
    
    def plot_threshold_evolution(self, save_path: str = "capla_threshold_evolution.png"):
        """Plot how adaptive thresholds evolve during training."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for class_id in range(len(self.class_names)):
            if self.threshold_history[class_id]:
                ax.plot(self.update_iterations, 
                       self.threshold_history[class_id],
                       marker='o',
                       label=f"Class {class_id}: {self.class_names[class_id]}",
                       linewidth=2)
        
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Adaptive Threshold ($t_c$)", fontsize=12)
        ax.set_title("CAPLA Threshold Evolution During Training", fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved threshold evolution plot to {save_path}")
        plt.close()
    
    def plot_score_distributions(self, save_path: str = "capla_score_distributions.png"):
        """Plot pseudolabel score distributions for each class."""
        num_classes = len(self.class_names)
        cols = 3
        rows = (num_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = axes.flatten()
        
        for class_id in range(num_classes):
            ax = axes[class_id]
            scores = self.score_distributions[class_id]
            
            if scores:
                ax.hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
                
                # Mark threshold
                if self.threshold_history[class_id]:
                    current_threshold = self.threshold_history[class_id][-1]
                    ax.axvline(current_threshold, color='red', linestyle='--', 
                              linewidth=2, label=f'Threshold: {current_threshold:.3f}')
                
                # Mark reliable/uncertain regions
                ax.axvspan(0.5, self.threshold_history[class_id][-1] if self.threshold_history[class_id] else 0.7,
                          alpha=0.2, color='orange', label='Uncertain')
                ax.axvspan(self.threshold_history[class_id][-1] if self.threshold_history[class_id] else 0.7, 1.0,
                          alpha=0.2, color='green', label='Reliable')
                
                ax.set_xlabel("Confidence Score", fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_title(f"Class {class_id}: {self.class_names[class_id]}", fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Class {class_id}: {self.class_names[class_id]}", fontsize=11, fontweight='bold')
        
        # Hide extra subplots
        for idx in range(num_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved score distributions plot to {save_path}")
        plt.close()
    
    def plot_reliable_uncertain_ratio(self, save_path: str = "capla_reliable_uncertain_ratio.png"):
        """Plot the ratio of reliable to uncertain pseudolabels per class."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class_ids = []
        reliable_counts = []
        uncertain_counts = []
        
        for class_id in range(len(self.class_names)):
            scores = self.score_distributions[class_id]
            if scores and self.threshold_history[class_id]:
                threshold = self.threshold_history[class_id][-1]
                scores_arr = np.array(scores)
                
                reliable = np.sum(scores_arr > threshold)
                uncertain = np.sum((scores_arr > 0.5) & (scores_arr <= threshold))
                
                if reliable + uncertain > 0:
                    class_ids.append(self.class_names[class_id])
                    reliable_counts.append(reliable)
                    uncertain_counts.append(uncertain)
        
        x = np.arange(len(class_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, reliable_counts, width, label='Reliable', color='green', alpha=0.8)
        bars2 = ax.bar(x + width/2, uncertain_counts, width, label='Uncertain', color='orange', alpha=0.8)
        
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Reliable vs Uncertain Pseudolabels per Class (CAPLA)", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved reliable/uncertain ratio plot to {save_path}")
        plt.close()
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("CAPLA ANALYSIS SUMMARY")
        print("="*60)
        
        for class_id in range(len(self.class_names)):
            class_name = self.class_names[class_id]
            scores = self.score_distributions[class_id]
            thresholds = self.threshold_history[class_id]
            
            if scores and thresholds:
                scores_arr = np.array(scores)
                current_threshold = thresholds[-1]
                initial_threshold = thresholds[0]
                
                reliable = np.sum(scores_arr > current_threshold)
                uncertain = np.sum((scores_arr > 0.5) & (scores_arr <= current_threshold))
                
                print(f"\nClass {class_id}: {class_name}")
                print(f"  Initial threshold: {initial_threshold:.3f}")
                print(f"  Final threshold:   {current_threshold:.3f}")
                print(f"  Change: {current_threshold - initial_threshold:+.3f}")
                print(f"  Reliable pseudolabels: {reliable}")
                print(f"  Uncertain pseudolabels: {uncertain}")
                print(f"  Score statistics (mean±std): {np.mean(scores):.3f}±{np.std(scores):.3f}")
                print(f"  Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
        
        print("\n" + "="*60)


def demonstrate_capla_formula():
    """
    Demonstrate the CAPLA threshold calculation formula.
    
    Formula: t_c = P_c^l * δ% * n_c^l * (N^u / N^l)
    """
    print("\n" + "="*60)
    print("CAPLA THRESHOLD FORMULA DEMONSTRATION")
    print("="*60)
    
    # Example parameters
    n_classes = 4
    class_names = {
        0: "scratch",
        1: "dent",
        2: "rivet-damage",
        3: "corrosion"
    }
    
    # Ground truth counts (simulated)
    gt_counts = {
        0: 500,    # Common class
        1: 450,    # Common class
        2: 80,     # Rare class
        3: 95      # Rare class
    }
    
    N_l = sum(gt_counts.values())  # Total labeled samples
    N_u = 5000  # Total unlabeled samples
    delta = 0.30  # 30% reliable threshold
    data_ratio = N_u / N_l
    
    # Simulated sorted scores for each class (30th percentile, P_c^l)
    percentile_scores = {
        0: 0.72,   # High quality predictions for common class
        1: 0.70,
        2: 0.58,   # Lower quality for rare class
        3: 0.61
    }
    
    print(f"\nDataset Configuration:")
    print(f"  Labeled samples (N^l): {N_l}")
    print(f"  Unlabeled samples (N^u): {N_u}")
    print(f"  Ratio (N^u/N^l): {data_ratio:.2f}")
    print(f"  Reliable threshold (δ%): {delta:.1%}")
    
    print(f"\nClass-Specific Thresholds:")
    print(f"{'Class':<20} {'GT Count':<10} {'P_c^l':<10} {'Formula':<50} {'t_c':<10}")
    print("-" * 100)
    
    thresholds = {}
    for class_id in range(n_classes):
        class_name = class_names[class_id]
        n_c_l = gt_counts[class_id]
        P_c_l = percentile_scores[class_id]
        
        # Apply CAPLA formula
        t_c = P_c_l * delta * n_c_l * data_ratio
        t_c = np.clip(t_c, 0.3, 0.95)  # Clip to reasonable range
        thresholds[class_id] = t_c
        
        formula = f"{P_c_l:.2f} × {delta:.2f} × {n_c_l} × {data_ratio:.2f}"
        print(f"{class_name:<20} {n_c_l:<10} {P_c_l:<10.2f} {formula:<50} {t_c:<10.3f}")
    
    print(f"\nInterpretation:")
    print(f"  Common classes (scratch, dent): Higher thresholds → Stricter filtering")
    print(f"  Rare classes (rivet-damage, corrosion): Lower thresholds → More pseudolabels")
    print(f"  This balances the pseudolabel distribution to match training data distribution")


def simulate_capla_training():
    """Simulate CAPLA training with realistic data."""
    print("\n" + "="*60)
    print("CAPLA TRAINING SIMULATION")
    print("="*60)
    
    class_names = {
        0: "scratch",
        1: "dent", 
        2: "rivet-damage",
        3: "corrosion"
    }
    
    analyzer = CAPLAAnalyzer(class_names)
    
    # Simulate training with 5 threshold updates
    initial_thresholds = {0: 0.72, 1: 0.70, 2: 0.58, 3: 0.61}
    
    for iteration in [0, 2000, 4000, 6000, 8000]:
        # Simulate threshold updates (thresholds increase as teacher improves)
        factor = iteration / 10000
        updated_thresholds = {
            class_id: min(0.95, initial + factor * 0.05)
            for class_id, initial in initial_thresholds.items()
        }
        
        analyzer.record_thresholds(iteration, updated_thresholds)
        
        # Simulate score distributions
        for class_id in range(4):
            if class_id in [0, 1]:  # Common classes
                scores = np.random.beta(8, 2, 500) * 0.5 + 0.5  # High scores
            else:  # Rare classes
                scores = np.random.beta(5, 3, 200) * 0.5 + 0.3  # Lower scores
            
            analyzer.record_scores(class_id, scores.tolist())
    
    # Generate visualizations
    print("\nGenerating CAPLA analysis visualizations...")
    analyzer.plot_threshold_evolution()
    analyzer.plot_score_distributions()
    analyzer.plot_reliable_uncertain_ratio()
    analyzer.print_summary()


if __name__ == "__main__":
    # Demonstrate CAPLA formula
    demonstrate_capla_formula()
    
    # Simulate CAPLA training
    simulate_capla_training()
    
    print("\n" + "="*60)
    print("CAPLA visualization complete!")
    print("Check the generated PNG files for detailed analysis:")
    print("  - capla_threshold_evolution.png")
    print("  - capla_score_distributions.png")
    print("  - capla_reliable_uncertain_ratio.png")
    print("="*60)
