"""
Report Generator for Attack Simulation

Generates comprehensive reports in multiple formats (HTML, Markdown, CSV, JSON).
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive attack simulation reports.
    """
    
    def __init__(self):
        """Initialize report generator."""
        logger.info("ReportGenerator initialized")
    
    def generate_html_report(
        self,
        attack_results: Dict[str, Dict],
        comparison_df: pd.DataFrame,
        per_class_df: Optional[pd.DataFrame],
        robustness_scores: Dict[str, float],
        visualization_paths: Dict[str, Path],
        output_path: Path
    ) -> None:
        """
        Generate beautiful HTML report with embedded visualizations.
        
        Args:
            attack_results: Dictionary of attack evaluation results
            comparison_df: Attack comparison DataFrame
            per_class_df: Per-class metrics DataFrame
            robustness_scores: Robustness scoring results
            visualization_paths: Paths to generated visualization images
            output_path: Path to save HTML report
        """
        logger.info(f"Generating HTML report to {output_path}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attack Simulation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-card h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .robustness-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .robustness-score {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .robustness-score .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .robustness-score .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .robustness-score.high {{
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        }}
        
        .robustness-score.medium {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        }}
        
        .robustness-score.low {{
            background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
            color: white;
        }}
        
        .robustness-score.low .value, .robustness-score.low .label {{
            color: white !important;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .visualization {{
            margin-top: 20px;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin: 10px 0;
        }}
        
        .footer {{
            background: #f5f7fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .badge-success {{
            background: #96e6a1;
            color: #2d3436;
        }}
        
        .badge-warning {{
            background: #fdcb6e;
            color: #2d3436;
        }}
        
        .badge-danger {{
            background: #ff7675;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Attack Simulation Report</h1>
            <div class="subtitle">Comprehensive Adversarial Attack Analysis</div>
            <div class="subtitle" style="margin-top: 10px; font-size: 0.9em;">
                Generated: {timestamp}
            </div>
        </div>
        
        <div class="content">
"""
        
        # Robustness Scores
        if robustness_scores:
            overall = robustness_scores.get('overall_robustness', 0)
            worst_case = robustness_scores.get('worst_case_robustness', 0)
            vulnerability = robustness_scores.get('vulnerability_score', 0)
            
            # Determine color class
            if overall >= 70:
                color_class = "high"
            elif overall >= 40:
                color_class = "medium"
            else:
                color_class = "low"
            
            html += f"""
            <div class="metric-card">
                <h2>📊 Model Robustness Assessment</h2>
                <div class="robustness-grid">
                    <div class="robustness-score {color_class}">
                        <div class="label">Overall Robustness</div>
                        <div class="value">{overall:.1f}</div>
                        <div class="label">out of 100</div>
                    </div>
                    <div class="robustness-score">
                        <div class="label">Best Case</div>
                        <div class="value">{robustness_scores.get('best_case_robustness', 0):.1f}</div>
                    </div>
                    <div class="robustness-score">
                        <div class="label">Worst Case</div>
                        <div class="value">{worst_case:.1f}</div>
                    </div>
                    <div class="robustness-score">
                        <div class="label">Vulnerability</div>
                        <div class="value">{vulnerability:.1f}%</div>
                    </div>
                </div>
            </div>
"""
        
        # Attack Comparison Table
        html += """
            <div class="metric-card">
                <h2>⚔️ Attack Effectiveness Comparison</h2>
                <table>
                    <tr>
                        <th>Attack Type</th>
                        <th>Effectiveness</th>
                        <th>Accuracy Drop</th>
                        <th>F1 Drop</th>
                        <th>Flip Rate</th>
                        <th>Evasion Success</th>
                    </tr>
"""
        
        for _, row in comparison_df.iterrows():
            effectiveness = row['effectiveness']
            badge_class = "badge-danger" if effectiveness > 0.5 else ("badge-warning" if effectiveness > 0.3 else "badge-success")
            
            html += f"""
                    <tr>
                        <td><strong>{row['attack']}</strong></td>
                        <td>{effectiveness:.3f} <span class="badge {badge_class}">
                            {'High' if effectiveness > 0.5 else ('Medium' if effectiveness > 0.3 else 'Low')}
                        </span></td>
                        <td>{row['accuracy_drop']:.4f}</td>
                        <td>{row['f1_drop']:.4f}</td>
                        <td>{row['prediction_flip_rate']:.4f}</td>
                        <td>{row['evasion_success_rate']:.4f}</td>
                    </tr>
"""
        
        html += """
                </table>
            </div>
"""
        
        # Per-Class Metrics (if available)
        if per_class_df is not None and not per_class_df.empty:
            html += """
            <div class="metric-card">
                <h2>📈 Per-Class Performance Impact</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Support</th>
                        <th>Accuracy Drop</th>
                        <th>Precision Drop</th>
                        <th>Recall Drop</th>
                        <th>F1 Drop</th>
                    </tr>
"""
            for _, row in per_class_df.iterrows():
                html += f"""
                    <tr>
                        <td><strong>{row['class']}</strong></td>
                        <td>{row['support']}</td>
                        <td>{row['accuracy_drop']:.4f}</td>
                        <td>{row['precision_drop']:.4f}</td>
                        <td>{row['recall_drop']:.4f}</td>
                        <td>{row['f1_drop']:.4f}</td>
                    </tr>
"""
            html += """
                </table>
            </div>
"""
        
        # Visualizations
        html += """
            <div class="metric-card">
                <h2>📊 Visual Analysis</h2>
"""
        
        for viz_name, viz_path in visualization_paths.items():
            if viz_path and viz_path.exists():
                html += f"""
                <div class="visualization">
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    <img src="{viz_path.name}" alt="{viz_name}">
                </div>
"""
        
        html += """
            </div>
"""
        
        # Close HTML
        html += f"""
        </div>
        
        <div class="footer">
            <p><strong>Hydra-IDS Attack Simulation Framework</strong> v1.0.0</p>
            <p>Report generated on {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def export_metrics_csv(
        self,
        comparison_df: pd.DataFrame,
        per_class_df: Optional[pd.DataFrame],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Export metrics to CSV files.
        
        Args:
            comparison_df: Attack comparison DataFrame
            per_class_df: Per-class metrics DataFrame
            output_dir: Output directory
            
        Returns:
            Dictionary mapping CSV names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_paths = {}
        
        # Export attack comparison
        comparison_path = output_dir / 'attack_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        csv_paths['comparison'] = comparison_path
        logger.info(f"Exported attack comparison to {comparison_path}")
        
        # Export per-class metrics if available
        if per_class_df is not None and not per_class_df.empty:
            per_class_path = output_dir / 'per_class_metrics.csv'
            per_class_df.to_csv(per_class_path, index=False)
            csv_paths['per_class'] = per_class_path
            logger.info(f"Exported per-class metrics to {per_class_path}")
        
        return csv_paths
    
    def generate_markdown_summary(
        self,
        comparison_df: pd.DataFrame,
        robustness_scores: Dict[str, float],
        output_path: Path
    ) -> None:
        """
        Generate Markdown summary report.
        
        Args:
            comparison_df: Attack comparison DataFrame
            robustness_scores: Robustness scoring results
            output_path: Path to save markdown file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# Attack Simulation Report

**Generated:** {timestamp}

## Model Robustness Score

"""
        
        if robustness_scores:
            overall = robustness_scores.get('overall_robustness', 0)
            md += f"""
- **Overall Robustness:** {overall:.1f}/100
- **Best Case:** {robustness_scores.get('best_case_robustness', 0):.1f}/100
- **Worst Case:** {robustness_scores.get('worst_case_robustness', 0):.1f}/100
- **Vulnerability Score:** {robustness_scores.get('vulnerability_score', 0):.1f}%

"""
        
        md += """## Attack Effectiveness Summary

| Attack | Effectiveness | Accuracy Drop | F1 Drop | Evasion Success |
|--------|---------------|---------------|---------|-----------------|
"""
        
        for _, row in comparison_df.iterrows():
            md += f"| {row['attack']} | {row['effectiveness']:.3f} | {row['accuracy_drop']:.4f} | {row['f1_drop']:.4f} | {row['evasion_success_rate']:.4f} |\n"
        
        md += """
## Interpretation

"""
        
        # Add interpretation based on results
        max_effectiveness = comparison_df['effectiveness'].max()
        if max_effectiveness > 0.5:
            md += "⚠️ **HIGH VULNERABILITY DETECTED**: Some attacks achieved high effectiveness (>0.5). Model requires hardening.\n\n"
        elif max_effectiveness > 0.3:
            md += "⚠️ **MODERATE VULNERABILITY**: Attacks showed moderate effectiveness. Consider defensive measures.\n\n"
        else:
            md += "✅ **GOOD ROBUSTNESS**: Model shows resilience against tested attacks.\n\n"
        
        # Save markdown
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        logger.info(f"Markdown summary saved to {output_path}")


if __name__ == "__main__":
    print("ReportGenerator module loaded successfully")
