"""
Enhanced Attack Simulation with Comprehensive Visualizations

Demonstrates the complete attack framework with:
- Multiple attack types
- Advanced metrics calculation
- Professional visualizations
- Comprehensive reporting (HTML, CSV, Markdown, JSON)

Run this script to generate a complete attack analysis report.
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
print("ENHANCED ATTACK SIMULATION - COMPREHENSIVE ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. Load Data and Model
# ============================================================================
print("\n[1/8] Loading data and model...")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'logs'
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
            # Extract class names in order
            label_names = [label_names_dict[str(i)] for i in sorted([int(k) for k in label_names_dict.keys()])]
        else:
            label_names = label_names_dict
    
    print(f"✓ Data shape: {X_test.shape}")
    print(f"✓ Labels: {len(label_names)} classes")
    print(f"✓ Model: {type(model).__name__}")
    
except Exception as e:
    logger.error(f"Error loading data/model: {e}")
    print(f"\n❌ Failed to load data or model: {e}")
    print("Please ensure the model is trained and data is processed.")
    sys.exit(1)

# Sample subset for faster testing
SAMPLE_SIZE = 5000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)

X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

print(f"✓ Using {len(X_sample)} samples for testing")

# ============================================================================
# 2. Initialize Attack Framework
# ============================================================================
print("\n[2/8] Initializing attack framework...")

attack_config = {
    'label_flip_fraction': 0.15,
    'feature_noise': 0.10,
    'drift_strength': 0.25,
    'drop_fraction': 0.20,
    'epsilon': 0.05,
    'missing_fraction': 0.10,
    'outlier_fraction': 0.05,
    'outlier_magnitude': 4.0,
    'random_state': 42
}

controller = AttackController(
    config=attack_config,
    model=model,
    scaler=scaler,
    track_history=True
)

controller.set_baseline(X_sample, y_sample)

# Initialize metrics and visualization
metrics_calc = AttackMetrics(model, scaler)
visualizer = AttackVisualizer(style='whitegrid', color_palette='Set2')
report_gen = ReportGenerator()

print("✓ Attack controller initialized")
print("✓ Metrics calculator ready")
print("✓ Visualizer initialized")
print("✓ Report generator ready")

# ============================================================================
# 3. Run Attack Scenarios
# ============================================================================
print("\n[3/8] Executing attack scenarios...")

attack_results = {}

# Scenario 1: Drift Attack
print("\n  → Running Drift Attack...")
X_drifted, drift_meta = controller.apply_drift(X_sample, evaluate=True, drift_type='gradual')
if 'evaluation' in drift_meta:
    attack_results['drift_gradual'] = drift_meta['evaluation']
    print(f"    Effectiveness: {drift_meta['evaluation']['attack_effectiveness']:.3f}")

# Scenario 2: Corruption Attack
print("\n  → Running Corruption Attack...")
X_corrupted, corruption_meta = controller.apply_corruption(
    X_sample, 
    evaluate=True, 
    corruption_type='drop_features'
)
if 'evaluation' in corruption_meta:
    attack_results['corruption_drop'] = corruption_meta['evaluation']
    print(f"    Effectiveness: {corruption_meta['evaluation']['attack_effectiveness']:.3f}")

# Scenario 3: Evasion Attack
print("\n  → Running Evasion Attack...")
X_evaded, evasion_meta = controller.apply_evasion(
    X_sample, 
    evaluate=True, 
    strategy='random_sign'
)
if 'evaluation' in evasion_meta:
    attack_results['evasion_random'] = evasion_meta['evaluation']
    print(f"    Effectiveness: {evasion_meta['evaluation']['attack_effectiveness']:.3f}")

# Scenario 4: Combined Attack Chain
print("\n  → Running Combined Attack Chain...")
X_clean, y_clean = controller.rollback()
X_chain, y_chain, chain_meta = controller.apply_attack_chain(
    X_clean,
    y_clean,
    attack_sequence=['drift', 'corruption', 'evasion']
)
if chain_meta and 'chain_evaluation' in chain_meta[-1]:
    attack_results['combined_chain'] = chain_meta[-1]['chain_evaluation']
    print(f"    Effectiveness: {chain_meta[-1]['chain_evaluation']['attack_effectiveness']:.3f}")

print(f"\n✓ Completed {len(attack_results)} attack scenarios")

# ============================================================================
# 4. Calculate Comprehensive Metrics
# ============================================================================
print("\n[4/8] Calculating comprehensive metrics...")

# Attack comparison
comparison_df = metrics_calc.compare_attacks(attack_results)
print("\n" + "="*70)
print("ATTACK COMPARISON SUMMARY")
print("="*70)
print(comparison_df.to_string(index=False))

# Robustness score
robustness_scores = metrics_calc.calculate_robustness_score(attack_results)
print(f"\n📊 Model Robustness Score: {robustness_scores['overall_robustness']:.1f}/100")
print(f"   Vulnerability: {robustness_scores['vulnerability_score']:.1f}%")

# Per-class metrics (using the most effective attack)
print("\n  → Calculating per-class metrics...")
most_effective_attack = comparison_df.iloc[0]['attack']  # Already sorted by effectiveness
most_effective_result = attack_results[most_effective_attack]

# Get predictions for per-class analysis
X_clean, y_clean = controller.rollback()
if scaler:
    X_clean_scaled = scaler.transform(X_clean)
else:
    X_clean_scaled = X_clean.values

y_pred_clean = model.predict(X_clean_scaled)

# Get attacked predictions from the most effective attack
if most_effective_attack in attack_results:
    attacked_performance = most_effective_result['attacked_performance']
    # We need to reconstruct predictions - use the last attack's data
    if most_effective_attack == 'combined_chain':
        X_attacked = X_chain
    elif most_effective_attack == 'drift_gradual':
        X_attacked = X_drifted
    elif most_effective_attack == 'corruption_drop':
        X_attacked = X_corrupted
    else:
        X_attacked = X_evaded
    
    if scaler:
        X_attacked_scaled = scaler.transform(X_attacked)
    else:
        X_attacked_scaled = X_attacked.values
    
    y_pred_attacked = model.predict(X_attacked_scaled)
    
    per_class_df = metrics_calc.calculate_per_class_metrics(
        y_clean, y_pred_clean, y_pred_attacked, label_names
    )
    print(f"✓ Per-class metrics calculated for {len(per_class_df)} classes")
else:
    per_class_df = None

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
    y_clean.values if hasattr(y_clean, 'values') else y_clean,
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

# 4. Feature Impact (from most effective attack)
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
    y_clean.values if hasattr(y_clean, 'values') else y_clean,
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
    y_clean.values if hasattr(y_clean, 'values') else y_clean,
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
    output_path=OUTPUT_DIR / 'attack_report.html'
)
print(f"    ✓ HTML: {OUTPUT_DIR / 'attack_report.html'}")

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
    OUTPUT_DIR / 'attack_summary.md'
)
print(f"    ✓ Markdown: {OUTPUT_DIR / 'attack_summary.md'}")

# JSON Report (existing)
print("  → Exporting JSON report...")
controller.export_report(OUTPUT_DIR / 'attack_report.json')
print(f"    ✓ JSON: {OUTPUT_DIR / 'attack_report.json'}")

# ============================================================================
# 7. Summary Statistics
# ============================================================================
print("\n[7/8] Generating summary...")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n📊 Attacks Simulated: {len(attack_results)}")
print(f"📈 Visualizations Created: {len(visualization_paths)}")
print(f"📝 Reports Generated: 4 (HTML, JSON, Markdown, CSV)")

print(f"\n🛡️  ROBUSTNESS ASSESSMENT")
print(f"   Overall Score: {robustness_scores['overall_robustness']:.1f}/100")
print(f"   Best Case:     {robustness_scores['best_case_robustness']:.1f}/100")
print(f"   Worst Case:    {robustness_scores['worst_case_robustness']:.1f}/100")
print(f"   Vulnerability: {robustness_scores['vulnerability_score']:.1f}%")

print(f"\n⚔️  MOST EFFECTIVE ATTACK")
most_effective_row = comparison_df.iloc[0]
print(f"   Type:              {most_effective_row['attack']}")
print(f"   Effectiveness:     {most_effective_row['effectiveness']:.3f}")
print(f"   Accuracy Drop:     {most_effective_row['accuracy_drop']:.4f}")
print(f"   Evasion Success:   {most_effective_row['evasion_success_rate']:.4f}")

print(f"\n📁 OUTPUT FILES:")
print(f"   HTML Report:  {OUTPUT_DIR / 'attack_report.html'}")
print(f"   Dashboard:    {OUTPUT_DIR / 'attack_dashboard.png'}")
print(f"   CSV Metrics:  {OUTPUT_DIR / 'attack_comparison.csv'}")
print(f"   Summary:      {OUTPUT_DIR / 'attack_summary.md'}")

# ============================================================================
# 8. Completion
# ============================================================================
print("\n[8/8] Complete!")

print("\n" + "=" * 70)
print("ATTACK SIMULATION SUCCESSFULLY COMPLETED")
print("=" * 70)

print(f"\n🎯 Next Steps:")
print(f"   1. Open the HTML report: {OUTPUT_DIR / 'attack_report.html'}")
print(f"   2. Review the attack dashboard: {OUTPUT_DIR / 'attack_dashboard.png'}")
print(f"   3. Analyze class-specific vulnerabilities in CSV exports")
print(f"   4. Implement defensive measures based on findings")

print("\n✅ All attack analysis artifacts saved to:", OUTPUT_DIR)
print()
