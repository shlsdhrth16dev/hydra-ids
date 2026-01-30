"""
Attack Visualization Module

Professional visualizations for attack metrics and outcomes.
Provides comprehensive visual analysis of adversarial attacks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


class AttackVisualizer:
    """
    Create professional visualizations for attack analysis.
    """
    
    def __init__(self, style: str = 'whitegrid', color_palette: str = 'Set2'):
        """
        Initialize visualizer.
        
        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark')
            color_palette: Color palette for plots
        """
        sns.set_style(style)
        self.color_palette = color_palette
        logger.info(f"AttackVisualizer initialized with style: {style}")
    
    def plot_attack_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Create comprehensive attack comparison visualization.
        
        Args:
            comparison_df: DataFrame from AttackMetrics.compare_attacks()
            output_path: Optional path to save the figure
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating attack comparison visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Attack Effectiveness Comparison', fontsize=16, fontweight='bold')
        
        # 1. Effectiveness Scores (Top Left)
        ax1 = axes[0, 0]
        comparison_df_sorted = comparison_df.sort_values('effectiveness', ascending=True)
        colors = sns.color_palette(self.color_palette, len(comparison_df))
        ax1.barh(comparison_df_sorted['attack'], comparison_df_sorted['effectiveness'], color=colors)
        ax1.set_xlabel('Effectiveness Score', fontweight='bold')
        ax1.set_ylabel('Attack Type', fontweight='bold')
        ax1.set_title('Overall Attack Effectiveness', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(comparison_df_sorted['effectiveness']):
            ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
        
        # 2. Model Degradation Metrics (Top Right)
        ax2 = axes[0, 1]
        metrics_to_plot = ['accuracy_drop', 'f1_drop']
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax2.bar(x - width/2, comparison_df['accuracy_drop'], width, 
                label='Accuracy Drop', alpha=0.8, color='#ff6b6b')
        ax2.bar(x + width/2, comparison_df['f1_drop'], width,
                label='F1 Drop', alpha=0.8, color='#4ecdc4')
        
        ax2.set_xlabel('Attack', fontweight='bold')
        ax2.set_ylabel('Performance Drop', fontweight='bold')
        ax2.set_title('Model Performance Degradation', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([a[:15] for a in comparison_df['attack']], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Prediction Changes (Bottom Left)
        ax3 = axes[1, 0]
        ax3.bar(comparison_df['attack'], comparison_df['prediction_flip_rate'],
                color='#95e1d3', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Attack Type', fontweight='bold')
        ax3.set_ylabel('Flip Rate', fontweight='bold')
        ax3.set_title('Prediction Flip Rate', fontweight='bold')
        ax3.set_xticklabels(comparison_df['attack'], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (attack, rate) in enumerate(zip(comparison_df['attack'], comparison_df['prediction_flip_rate'])):
            ax3.text(i, rate + 0.01, f'{rate:.3f}', ha='center', fontsize=8)
        
        # 4. Evasion Success Rate (Bottom Right)
        ax4 = axes[1, 1]
        ax4.bar(comparison_df['attack'], comparison_df['evasion_success_rate'],
                color='#f38181', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Attack Type', fontweight='bold')
        ax4.set_ylabel('Evasion Success Rate', fontweight='bold')
        ax4.set_title('Attack Evasion Success', fontweight='bold')
        ax4.set_xticklabels(comparison_df['attack'], rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (attack, rate) in enumerate(zip(comparison_df['attack'], comparison_df['evasion_success_rate'])):
            ax4.text(i, rate + 0.01, f'{rate:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Attack comparison saved to {output_path}")
        
        return fig
    
    def plot_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred_clean: np.ndarray,
        y_pred_attacked: np.ndarray,
        class_names: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 5)
    ) -> plt.Figure:
        """
        Plot confusion matrices before and after attack.
        
        Args:
            y_true: True labels
            y_pred_clean: Predictions on clean data
            y_pred_attacked: Predictions on attacked data
            class_names: Optional class names for labels
            output_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating confusion matrix comparison...")
        
        # Calculate confusion matrices
        cm_clean = confusion_matrix(y_true, y_pred_clean)
        cm_attacked = confusion_matrix(y_true, y_pred_attacked)
        cm_diff = cm_attacked - cm_clean
        
        # Determine the number of classes from the confusion matrix
        n_classes = cm_clean.shape[0]
        
        # Handle class names safely
        if class_names is None or len(class_names) != n_classes:
            class_names = [f'C{i}' for i in range(n_classes)]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Confusion Matrix Analysis: Before vs After Attack', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Clean predictions
        sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        axes[0].set_title('Before Attack (Clean)', fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        
        # Plot 2: Attacked predictions
        sns.heatmap(cm_attacked, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        axes[1].set_title('After Attack', fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        
        # Plot 3: Difference (what changed)
        sns.heatmap(cm_diff, annot=True, fmt='d', cmap='RdYlGn_r', center=0, ax=axes[2],
                   xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Change'})
        axes[2].set_title('Difference (Attack Impact)', fontweight='bold')
        axes[2].set_ylabel('True Label', fontweight='bold')
        axes[2].set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {output_path}")
        
        return fig
    
    def plot_temporal_degradation(
        self,
        attack_history: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot model performance degradation over attack sequence.
        
        Args:
            attack_history: List of attack metadata with evaluations
            output_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating temporal degradation plot...")
        
        # Extract metrics over time
        timestamps = []
        accuracy_clean = []
        accuracy_attacked = []
        f1_clean = []
        f1_attacked = []
        attack_names = []
        
        for i, attack in enumerate(attack_history):
            if 'evaluation' in attack:
                eval_data = attack['evaluation']
                timestamps.append(i)
                accuracy_clean.append(eval_data['clean_performance']['accuracy'])
                accuracy_attacked.append(eval_data['attacked_performance']['accuracy'])
                f1_clean.append(eval_data['clean_performance']['f1_weighted'])
                f1_attacked.append(eval_data['attacked_performance']['f1_weighted'])
                
                # Get attack name
                attack_name = attack.get('attack_type', attack.get('attack_combination', 'unknown'))
                attack_names.append(attack_name)
        
        if not timestamps:
            logger.warning("No evaluation data found in attack history")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle('Model Performance Degradation Over Attack Sequence', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Accuracy
        ax1 = axes[0]
        ax1.plot(timestamps, accuracy_clean, 'o-', label='Clean Data', 
                color='#2ecc71', linewidth=2, markersize=8)
        ax1.plot(timestamps, accuracy_attacked, 's-', label='After Attack', 
                color='#e74c3c', linewidth=2, markersize=8)
        ax1.fill_between(timestamps, accuracy_clean, accuracy_attacked, 
                         alpha=0.2, color='#e74c3c')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy Degradation', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: F1 Score
        ax2 = axes[1]
        ax2.plot(timestamps, f1_clean, 'o-', label='Clean Data',
                color='#3498db', linewidth=2, markersize=8)
        ax2.plot(timestamps, f1_attacked, 's-', label='After Attack',
                color='#e67e22', linewidth=2, markersize=8)
        ax2.fill_between(timestamps, f1_clean, f1_attacked,
                         alpha=0.2, color='#e67e22')
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.set_xlabel('Attack Sequence', fontweight='bold')
        ax2.set_title('F1 Score Degradation', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(timestamps)
        ax2.set_xticklabels(attack_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Temporal degradation plot saved to {output_path}")
        
        return fig
    
    def plot_feature_impact(
        self,
        attack_metadata: Dict[str, Any],
        top_n: int = 20,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize which features were most impacted by attacks.
        
        Args:
            attack_metadata: Attack metadata containing feature information
            top_n: Number of top features to display
            output_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating feature impact visualization...")
        
        # Extract feature information based on attack type
        feature_data = []
        
        if 'drifted_features' in attack_metadata and 'shift_vector' in attack_metadata:
            # Drift attack
            for feature, shift in attack_metadata['shift_vector'].items():
                feature_data.append({
                    'feature': feature,
                    'impact': abs(shift),
                    'direction': 'positive' if shift > 0 else 'negative'
                })
        elif 'corrupted_features' in attack_metadata:
            # Corruption attack
            for feature in attack_metadata['corrupted_features']:
                feature_data.append({
                    'feature': feature,
                    'impact': 1.0,  # Binary impact
                    'direction': 'corrupted'
                })
        elif 'perturbed_features' in attack_metadata and 'perturbation_statistics' in attack_metadata:
            # Evasion attack - assign equal impact to all perturbed features
            avg_impact = attack_metadata['perturbation_statistics'].get('std', 0.05)
            for feature in attack_metadata['perturbed_features'][:top_n]:
                feature_data.append({
                    'feature': feature,
                    'impact': avg_impact,
                    'direction': 'perturbed'
                })
        
        if not feature_data:
            logger.warning("No feature impact data found in metadata")
            return None
        
        # Create DataFrame and sort
        df = pd.DataFrame(feature_data)
        df = df.nlargest(top_n, 'impact')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by direction
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 
                 'corrupted': '#9b59b6', 'perturbed': '#f39c12'}
        color_list = [colors.get(d, '#95a5a6') for d in df['direction']]
        
        # Create horizontal bar chart
        bars = ax.barh(df['feature'], df['impact'], color=color_list, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Impact Magnitude', fontweight='bold')
        ax.set_ylabel('Feature', fontweight='bold')
        ax.set_title(f'Top {top_n} Features Affected by Attack', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (feature, impact) in enumerate(zip(df['feature'], df['impact'])):
            ax.text(impact, i, f' {impact:.4f}', va='center', fontsize=8)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label.capitalize()) 
                          for label, color in colors.items() if label in df['direction'].values]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature impact plot saved to {output_path}")
        
        return fig
    
    def plot_class_performance(
        self,
        y_true: np.ndarray,
        y_pred_clean: np.ndarray,
        y_pred_attacked: np.ndarray,
        class_names: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Compare per-class performance before and after attack.
        
        Args:
            y_true: True labels
            y_pred_clean: Predictions on clean data
            y_pred_attacked: Predictions on attacked data
            class_names: Optional class names
            output_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating class-wise performance comparison...")
        
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate per-class metrics
        precision_clean, recall_clean, f1_clean, support = precision_recall_fscore_support(
            y_true, y_pred_clean, average=None, zero_division=0
        )
        precision_attacked, recall_attacked, f1_attacked, _ = precision_recall_fscore_support(
            y_true, y_pred_attacked, average=None, zero_division=0
        )
        
        # Prepare data
        n_classes = len(precision_clean)
        
        # Handle class names safely
        if class_names is None or len(class_names) != n_classes:
            class_names = [f'C{i}' for i in range(n_classes)]
        
        x = np.arange(n_classes)
        width = 0.35
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Per-Class Performance: Before vs After Attack', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Precision
        ax1 = axes[0]
        ax1.bar(x - width/2, precision_clean, width, label='Clean', 
               alpha=0.8, color='#3498db')
        ax1.bar(x + width/2, precision_attacked, width, label='Attacked',
               alpha=0.8, color='#e74c3c')
        ax1.set_ylabel('Precision', fontweight='bold')
        ax1.set_title('Precision by Class', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Recall
        ax2 = axes[1]
        ax2.bar(x - width/2, recall_clean, width, label='Clean',
               alpha=0.8, color='#2ecc71')
        ax2.bar(x + width/2, recall_attacked, width, label='Attacked',
               alpha=0.8, color='#e67e22')
        ax2.set_ylabel('Recall', fontweight='bold')
        ax2.set_title('Recall by Class', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: F1 Score
        ax3 = axes[2]
        ax3.bar(x - width/2, f1_clean, width, label='Clean',
               alpha=0.8, color='#9b59b6')
        ax3.bar(x + width/2, f1_attacked, width, label='Attacked',
               alpha=0.8, color='#f39c12')
        ax3.set_ylabel('F1 Score', fontweight='bold')
        ax3.set_title('F1 Score by Class', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Class performance plot saved to {output_path}")
        
        return fig
    
    def create_dashboard(
        self,
        comparison_df: pd.DataFrame,
        attack_history: List[Dict[str, Any]],
        y_true: np.ndarray,
        y_pred_clean: np.ndarray,
        y_pred_attacked: np.ndarray,
        class_names: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (20, 12)
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            comparison_df: Attack comparison DataFrame
            attack_history: Attack history with evaluations
            y_true: True labels
            y_pred_clean: Clean predictions
            y_pred_attacked: Attacked predictions
            class_names: Optional class names
            output_path: Optional path to save
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating comprehensive attack dashboard...")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Attack Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # Top row: Attack comparison (2 cols) + Temporal (1 col)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Middle row: Confusion matrices (3 cols)
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Bottom row: Class performance (3 cols)
        ax6 = fig.add_subplot(gs[2, 0])
        ax7 = fig.add_subplot(gs[2, 1])
        ax8 = fig.add_subplot(gs[2, 2])
        
        # 1. Attack effectiveness comparison
        comparison_df_sorted = comparison_df.sort_values('effectiveness', ascending=True)
        colors = sns.color_palette(self.color_palette, len(comparison_df))
        ax1.barh(comparison_df_sorted['attack'], comparison_df_sorted['effectiveness'], color=colors)
        ax1.set_xlabel('Effectiveness Score', fontweight='bold')
        ax1.set_title('Attack Effectiveness', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Evasion success
        ax2.bar(comparison_df['attack'], comparison_df['evasion_success_rate'],
               color='#f38181', alpha=0.8)
        ax2.set_ylabel('Evasion Rate', fontweight='bold')
        ax2.set_title('Evasion Success', fontweight='bold')
        ax2.set_xticklabels(comparison_df['attack'], rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3-5. Confusion matrices
        cm_clean = confusion_matrix(y_true, y_pred_clean)
        cm_attacked = confusion_matrix(y_true, y_pred_attacked)
        cm_diff = cm_attacked - cm_clean
        
        # Determine number of classes and fix class names
        n_classes_cm = cm_clean.shape[0]
        if class_names is None or len(class_names) != n_classes_cm:
            cm_class_names = [f'C{i}' for i in range(n_classes_cm)]
        else:
            cm_class_names = class_names
        
        sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=cm_class_names, yticklabels=cm_class_names, cbar=False)
        ax3.set_title('Before Attack', fontweight='bold')
        
        sns.heatmap(cm_attacked, annot=True, fmt='d', cmap='Oranges', ax=ax4,
                   xticklabels=cm_class_names, yticklabels=cm_class_names, cbar=False)
        ax4.set_title('After Attack', fontweight='bold')
        
        sns.heatmap(cm_diff, annot=True, fmt='d', cmap='RdYlGn_r', center=0, ax=ax5,
                   xticklabels=cm_class_names, yticklabels=cm_class_names, cbar=False)
        ax5.set_title('Impact', fontweight='bold')
        
        # 6-8. Class performance
        from sklearn.metrics import precision_recall_fscore_support
        precision_clean, recall_clean, f1_clean, _ = precision_recall_fscore_support(
            y_true, y_pred_clean, average=None, zero_division=0
        )
        precision_attacked, recall_attacked, f1_attacked, _ = precision_recall_fscore_support(
            y_true, y_pred_attacked, average=None, zero_division=0
        )
        
        n_classes = len(precision_clean)
        
        # Handle class names safely for bar plots
        if class_names is None or len(class_names) != n_classes:
            bar_class_names = [f'C{i}' for i in range(n_classes)]
        else:
            bar_class_names = class_names
        
        x = np.arange(n_classes)
        width = 0.35
        
        ax6.bar(x - width/2, precision_clean, width, label='Clean', alpha=0.8, color='#3498db')
        ax6.bar(x + width/2, precision_attacked, width, label='Attacked', alpha=0.8, color='#e74c3c')
        ax6.set_ylabel('Precision', fontweight='bold')
        ax6.set_title('Precision', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(bar_class_names, rotation=45, ha='right', fontsize=8)
        ax6.legend(fontsize=8)
        ax6.grid(axis='y', alpha=0.3)
        
        ax7.bar(x - width/2, recall_clean, width, label='Clean', alpha=0.8, color='#2ecc71')
        ax7.bar(x + width/2, recall_attacked, width, label='Attacked', alpha=0.8, color='#e67e22')
        ax7.set_ylabel('Recall', fontweight='bold')
        ax7.set_title('Recall', fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(bar_class_names, rotation=45, ha='right', fontsize=8)
        ax7.legend(fontsize=8)
        ax7.grid(axis='y', alpha=0.3)
        
        ax8.bar(x - width/2, f1_clean, width, label='Clean', alpha=0.8, color='#9b59b6')
        ax8.bar(x + width/2, f1_attacked, width, label='Attacked', alpha=0.8, color='#f39c12')
        ax8.set_ylabel('F1 Score', fontweight='bold')
        ax8.set_title('F1 Score', fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(bar_class_names, rotation=45, ha='right', fontsize=8)
        ax8.legend(fontsize=8)
        ax8.grid(axis='y', alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Dashboard saved to {output_path}")
        
        return fig


if __name__ == "__main__":
    print("AttackVisualizer module loaded successfully")
