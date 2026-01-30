"""
Attack Simulation - STRESS TEST

This version uses AGGRESSIVE attack parameters to demonstrate visible
model degradation and showcase the visualization/reporting capabilities.

⚠️ WARNING: These attack strengths are UNREALISTICALLY HIGH and are only
for testing the framework's visualization and reporting features.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Import attack framework
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from attacks import AttackController, AttackMetrics, AttackVisualizer
from attacks.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("STRESS TEST - AGGRESSIVE ATTACK SIMULATION")
print("=" * 70)
print("\n⚠️  Using UNREALISTICALLY STRONG attacks for demonstration purposes")
print("="  * 70)

# ============================================================================
# 1. Load Data and Model
# ============================================================================
print("\n[1/8] Loading data and model...")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'logs' / 'stress_test'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    # Load model and preprocessing
    model = joblib.load(MODEL_DIR / 'xgboost/xgboost_v1.joblib')
    scaler = joblib.load(MODEL_DIR / 'preprocessing/scaler.joblib')
    
    # Load label names
    with open(MODEL_DIR / 'preprocessing/label_names.json', 'r') as f:
        label_names_dict = json.load(f)
        # Handle both dict and list formats
        if isinstance(label_names_dict, dict):
            label_names = [label_names_dict[str(i)] for i in sorted([int(k) for k in label_names_dict.keys()])]
        else:
            label_names = label_names_dict
    
    print(f"✓ Data shape: {X_test.shape}")
    print(f"✓ Labels: {len(label_names)} classes")
    print(f"✓ Model: {type(model).__name__}")
    
except Exception as e:
    logger.error(f"Error loading data/model: {e}")
    print(f"\n❌ Failed to load data or model: {e}")
    sys.exit(1)

# Sample subset
SAMPLE_SIZE = 5000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)

X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

print(f"✓ Using {len(X_sample)} samples for testing")

# ============================================================================
# 2. Initialize with AGGRESSIVE Configuration
# ============================================================================
print("\n[2/8] Initializing with AGGRESSIVE attack parameters...")

# 🔥 STRESS TEST CONFIGURATION - Unrealistically high values
aggressive_config = {
    'label_flip_fraction': 0.30,    # 30% label flipping (vs 15% normal)
    'feature_noise': 0.25,          # 25% noise (vs 10% normal)
    'drift_strength': 0.75,         # 75% drift (vs 25% normal)
    'drop_fraction': 0.40,          # Drop 40% features (vs 20% normal)
    'epsilon': 0.30,                # 30% evasion (vs 5% normal)
    'missing_fraction': 0.25,       # 25% missing (vs 10% normal)
    'outlier_fraction': 0.15,       # 15% outliers (vs 5% normal)
    'outlier_magnitude': 8.0,       # Very strong outliers
    'random_state': 42
}

print("\n🔥 AGGRESSIVE ATTACK CONFIGURATION:")
for key, value in aggressive_config.items():
    if key != 'random_state':
        print(f"   {key}: {value}")

controller = AttackController(
    config=aggressive_config,
    model=model,
    scaler=scaler,
    track_history=True
)

controller.set_baseline(X_sample, y_sample)

# Initialize metrics and visualization
metrics_calc = AttackMetrics(model, scaler)
visualizer = AttackVisualizer(style='whitegrid', color_palette='Set2')
report_gen = ReportGenerator()

print("\n✓ Attack controller initialized")
print("✓ Metrics calculator ready")
print("✓ Visualizer initialized")

# ============================================================================
# 3. Run Attack Scenarios
# ============================================================================
print("\n[3/8] Executing AGGRESSIVE attack scenarios...")

attack_results = {}

# Scenario 1: Strong Drift Attack
print("\n  → Running STRONG Drift Attack (75% strength)...")
X_drifted, drift_meta = controller.apply_drift(X_sample, evaluate=True, drift_type='gradual')
if 'evaluation' in drift_meta:
    attack_results['drift_strong'] = drift_meta['evaluation']
    print(f"    Effectiveness: {drift_meta['evaluation']['attack_effectiveness']:.3f}")
    print(f"    Accuracy Drop: {drift_meta['evaluation']['model_degradation']['accuracy_drop']:.4f}")

# Scenario 2: Strong Corruption Attack
print("\n  → Running STRONG Corruption Attack (40% feature drop)...")
X_corrupted, corruption_meta = controller.apply_corruption(
    X_sample, 
    evaluate=True, 
    corruption_type='drop_features'
)
if 'evaluation' in corruption_meta:
    attack_results['corruption_strong'] = corruption_meta['evaluation']
    print(f"    Effectiveness: {corruption_meta['evaluation']['attack_effectiveness']:.3f}")
    print(f"    Accuracy Drop: {corruption_meta['evaluation']['model_degradation']['accuracy_drop']:.4f}")

# Scenario 3: Strong Evasion Attack
print("\n  → Running STRONG Evasion Attack (30% perturbation)...")
X_evaded, evasion_meta = controller.apply_evasion(
    X_sample, 
    evaluate=True, 
    strategy='random_sign'
)
if 'evaluation' in evasion_meta:
    attack_results['evasion_strong'] = evasion_meta['evaluation']
    print(f"    Effectiveness: {evasion_meta['evaluation']['attack_effectiveness']:.3f}")
    print(f"    Accuracy Drop: {evasion_meta['evaluation']['model_degradation']['accuracy_drop']:.4f}")

# Scenario 4: Combined Attack Chain (NO ROLLBACK - FIX FOR BUG)
print("\n  → Running Combined STRONG Attack Chain...")
# DON'T use rollback() - it clears history!
# Use baseline copies directly
X_clean = X_sample.copy()
y_clean = y_sample.copy()

X_chain, y_chain, chain_meta = controller.apply_attack_chain(
    X_clean,
    y_clean,
    attack_sequence=['drift', 'corruption', 'evasion']
)
if chain_meta and 'chain_evaluation' in chain_meta[-1]:
    attack_results['combined_strong'] = chain_meta[-1]['chain_evaluation']
    print(f"    Effectiveness: {chain_meta[-1]['chain_evaluation']['attack_effectiveness']:.3f}")
    print(f"    Accuracy Drop: {chain_meta[-1]['chain_evaluation']['model_degradation']['accuracy_drop']:.4f}")

print(f"\n✓ Completed {len(attack_results)} attack scenarios")

# ============================================================================
# 4. Calculate Comprehensive Metrics
# ============================================================================
print("\n[4/8] Calculating comprehensive metrics...")

# Attack comparison
comparison_df = metrics_calc.compare_attacks(attack_results)
print("\n" + "="*70)
print("STRESS TEST ATTACK COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))

# Robustness score
robustness_scores = metrics_calc.calculate_robustness_score(attack_results)
print(f"\n📊 Model Robustness Score: {robustness_scores['overall_robustness']:.1f}/100")
print(f"   Vulnerability: {robustness_scores['vulnerability_score']:.1f}%")

# Per-class metrics
print("\n  → Calculating per-class metrics...")
most_effective_attack = comparison_df.iloc[0]['attack']
most_effective_result = attack_results[most_effective_attack]

# Get baseline and attacked data
X_baseline, y_baseline = controller.original_data
if scaler:
    X_baseline_scaled = scaler.transform(X_baseline)
else:
    X_baseline_scaled = X_baseline.values

y_pred_clean = model.predict(X_baseline_scaled)

# Get attacked predictions
if most_effective_attack == 'combined_strong':
    X_attacked = X_chain
elif most_effective_attack == 'drift_strong':
    X_attacked = X_drifted
elif most_effective_attack == 'corruption_strong':
    X_attacked = X_corrupted
else:
    X_attacked = X_evaded

if scaler:
    X_attacked_scaled = scaler.transform(X_attacked)
else:
    X_attacked_scaled = X_attacked.values

y_pred_attacked = model.predict(X_attacked_scaled)

per_class_df = metrics_calc.calculate_per_class_metrics(
    y_baseline, y_pred_clean, y_pred_attacked, label_names
)
print(f"✓ Per-class metrics calculated for {len(per_class_df)} classes")

# ============================================================================
# 5. Generate Visualizations
# ============================================================================
print("\n[5/8] Creating visualizations...")

visualization_paths = {}

# 1. Attack Comparison
print("  → Attack comparison plot...")
fig1 = visualizer.plot_attack_comparison(
    comparison_df,
    output_path=OUTPUT_DIR / 'attack_comparison.png'
)
visualization_paths['Attack Comparison'] = OUTPUT_DIR / 'attack_comparison.png'
plt.close(fig1)

# 2. Confusion Matrices
print("  → Confusion matrices...")
fig2 = visualizer.plot_confusion_matrices(
    y_baseline.values if hasattr(y_baseline, 'values') else y_baseline,
    y_pred_clean,
    y_pred_attacked,
    class_names=label_names,
    output_path=OUTPUT_DIR / 'confusion_matrices.png'
)
visualization_paths['Confusion Matrices'] = OUTPUT_DIR / 'confusion_matrices.png'
plt.close(fig2)

# 3. Temporal Degradation
print("  → Temporal degradation...")
fig3 = visualizer.plot_temporal_degradation(
    controller.attack_history,
    output_path=OUTPUT_DIR / 'temporal_degradation.png'
)
if fig3:
    visualization_paths['Temporal Degradation'] = OUTPUT_DIR / 'temporal_degradation.png'
    plt.close(fig3)

# 4. Feature Impact
print("  → Feature impact analysis...")
for attack_info in controller.attack_history:
    if attack_info.get('attack_type') == most_effective_attack.split('_')[0]:
        fig4 = visualizer.plot_feature_impact(
            attack_info,
            top_n=20,
            output_path=OUTPUT_DIR / 'feature_impact.png'
        )
        if fig4:
            visualization_paths['Feature Impact'] = OUTPUT_DIR / 'feature_impact.png'
            plt.close(fig4)
        break

# 5. Class Performance
print("  → Per-class performance...")
fig5 = visualizer.plot_class_performance(
    y_baseline.values if hasattr(y_baseline, 'values') else y_baseline,
    y_pred_clean,
    y_pred_attacked,
    class_names=label_names,
    output_path=OUTPUT_DIR / 'class_performance.png'
)
visualization_paths['Class Performance'] = OUTPUT_DIR / 'class_performance.png'
plt.close(fig5)

# 6. Comprehensive Dashboard
print("  → Creating comprehensive dashboard...")
fig6 = visualizer.create_dashboard(
    comparison_df,
    controller.attack_history,
    y_baseline.values if hasattr(y_baseline, 'values') else y_baseline,
    y_pred_clean,
    y_pred_attacked,
    class_names=label_names,
    output_path=OUTPUT_DIR / 'attack_dashboard.png'
)
visualization_paths['Attack Dashboard'] = OUTPUT_DIR / 'attack_dashboard.png'
plt.close(fig6)

print(f"✓ Created {len(visualization_paths)} visualizations")

# ============================================================================
# 6. Generate Reports
# ============================================================================
print("\n[6/8] Generating reports...")

# HTML Report
print("  → Generating HTML report...")
report_gen.generate_html_report(
    attack_results=attack_results,
    comparison_df=comparison_df,
    per_class_df=per_class_df,
    robustness_scores=robustness_scores,
    visualization_paths=visualization_paths,
    output_path=OUTPUT_DIR / 'stress_test_report.html'
)
print(f"    ✓ HTML: {OUTPUT_DIR / 'stress_test_report.html'}")

# CSV Exports
print("  → Exporting metrics to CSV...")
csv_paths = report_gen.export_metrics_csv(
    comparison_df,
    per_class_df,
    OUTPUT_DIR
)
for name, path in csv_paths.items():
    print(f"    ✓ CSV ({name}): {path}")

# Markdown Summary
print("  → Generating Markdown summary...")
report_gen.generate_markdown_summary(
    comparison_df,
    robustness_scores,
    OUTPUT_DIR / 'stress_test_summary.md'
)
print(f"    ✓ Markdown: {OUTPUT_DIR / 'stress_test_summary.md'}")

# JSON Report
print("  → Exporting JSON report...")
controller.export_report(OUTPUT_DIR / 'stress_test_report.json')
print(f"    ✓ JSON: {OUTPUT_DIR / 'stress_test_report.json'}")

# ============================================================================
# 7. Summary Statistics
# ============================================================================
print("\n[7/8] Generating summary...")

print("\n" + "=" * 70)
print("STRESS TEST RESULTS")
print("=" * 70)

print(f"\n📊 Attacks Simulated: {len(attack_results)}")
print(f"📈 Visualizations Created: {len(visualization_paths)}")
print(f"📝 Reports Generated: 4 (HTML, JSON, Markdown, CSV)")

print(f"\n🛡️  ROBUSTNESS UNDER EXTREME STRESS")
print(f"   Overall Score: {robustness_scores['overall_robustness']:.1f}/100")
print(f"   Best Case:     {robustness_scores['best_case_robustness']:.1f}/100")
print(f"   Worst Case:    {robustness_scores['worst_case_robustness']:.1f}/100")
print(f"   Vulnerability: {robustness_scores['vulnerability_score']:.1f}%")

print(f"\n⚔️  MOST DAMAGING ATTACK")
most_effective_row = comparison_df.iloc[0]
print(f"   Type:              {most_effective_row['attack']}")
print(f"   Effectiveness:     {most_effective_row['effectiveness']:.3f}")
print(f"   Accuracy Drop:     {most_effective_row['accuracy_drop']:.4f}")
print(f"   F1 Drop:           {most_effective_row['f1_drop']:.4f}")
print(f"   Evasion Success:   {most_effective_row['evasion_success_rate']:.4f}")

print(f"\n📁 OUTPUT FILES:")
print(f"   HTML Report:  {OUTPUT_DIR / 'stress_test_report.html'}")
print(f"   Dashboard:    {OUTPUT_DIR / 'attack_dashboard.png'}")
print(f"   CSV Metrics:  {OUTPUT_DIR / 'attack_comparison.csv'}")
print(f"   Summary:      {OUTPUT_DIR / 'stress_test_summary.md'}")

# ============================================================================
# 8. Completion
# ============================================================================
print("\n[8/8] Complete!")

print("\n" + "=" * 70)
print("STRESS TEST SUCCESSFULLY COMPLETED")
print("=" * 70)

print(f"\n⚠️  IMPORTANT NOTES:")
print(f"   • These attack strengths are UNREALISTICALLY HIGH")
print(f"   • Used for framework demonstration purposes only")
print(f"   • Real-world attacks would use standard configuration")
print(f"   • Refer to data/logs/ for realistic attack results")

print(f"\n🎯 Next Steps:")
print(f"   1. Open HTML report: {OUTPUT_DIR / 'stress_test_report.html'}")
print(f"   2. Compare with realistic results in data/logs/")
print(f"   3. Review visualizations showing actual degradation")

print(f"\n✅ All stress test artifacts saved to: {OUTPUT_DIR}")
print()
