"""Report generator for evaluation results."""

import logging
from pathlib import Path
from datetime import datetime
from typing import TextIO
import sys

from app.evaluation.schemas import EvaluationReport, SystemEvaluation, EvalMetrics

logger = logging.getLogger(__name__)

# Default output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"


class ReportGenerator:
    """
    Generate human-readable reports from evaluation results.
    
    Supports console output and markdown file generation.
    """
    
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save report files
        """
        self.output_dir = Path(output_dir)
    
    def print_summary(self, report: EvaluationReport, file: TextIO = sys.stdout) -> None:
        """
        Print a summary of results to console.
        
        Args:
            report: The evaluation report
            file: Output file (default: stdout)
        """
        print(f"\n{'='*60}", file=file)
        print(f"EVALUATION RESULTS", file=file)
        print(f"Dataset: {report.dataset_name} (n={report.dataset_size})", file=file)
        print(f"Timestamp: {report.timestamp}", file=file)
        print(f"{'='*60}\n", file=file)
        
        # Summary table
        print(f"{'System':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Avg Cost':>12} {'Avg Latency':>14}", file=file)
        print("-" * 75, file=file)
        
        for name, eval in sorted(report.evaluations.items()):
            m = eval.metrics
            print(
                f"{name:<25} {m.accuracy:>9.1%} {m.macro_f1:>10.3f} "
                f"${m.avg_cost_usd:>10.5f} {m.avg_latency_ms:>11.0f}ms",
                file=file
            )
        
        print("-" * 75, file=file)
        
        # Best system
        best_accuracy = max(report.evaluations.items(), key=lambda x: x[1].metrics.accuracy)
        best_cost = min(report.evaluations.items(), key=lambda x: x[1].metrics.avg_cost_usd)
        
        print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1].metrics.accuracy:.1%})", file=file)
        print(f"Lowest Cost: {best_cost[0]} (${best_cost[1].metrics.avg_cost_usd:.5f}/query)", file=file)
        
        # Per-class breakdown for best system
        print(f"\n--- Per-Class Metrics ({best_accuracy[0]}) ---", file=file)
        for label, cm in best_accuracy[1].metrics.class_metrics.items():
            print(f"  {label:>6}: P={cm.precision:.3f} R={cm.recall:.3f} F1={cm.f1:.3f} (n={cm.support})", file=file)
    
    def generate_markdown(self, report: EvaluationReport) -> str:
        """
        Generate a markdown report.
        
        Args:
            report: The evaluation report
            
        Returns:
            Markdown string
        """
        lines = []
        
        # Header
        lines.append(f"# Evaluation Report")
        lines.append(f"")
        lines.append(f"**Dataset:** {report.dataset_name}")
        lines.append(f"**Sample Size:** {report.dataset_size}")
        lines.append(f"**Timestamp:** {report.timestamp}")
        lines.append(f"")
        
        # Summary table
        lines.append(f"## Summary")
        lines.append(f"")
        lines.append(f"| System | Accuracy | Macro-F1 | Avg Cost | Avg Latency | Tokens/Query |")
        lines.append(f"|--------|----------|----------|----------|-------------|--------------|")
        
        for name, eval in sorted(report.evaluations.items()):
            m = eval.metrics
            lines.append(
                f"| {name} | {m.accuracy:.1%} | {m.macro_f1:.3f} | "
                f"${m.avg_cost_usd:.5f} | {m.avg_latency_ms:.0f}ms | {m.avg_tokens:.0f} |"
            )
        
        lines.append(f"")
        
        # Detailed metrics per system
        lines.append(f"## Detailed Results")
        lines.append(f"")
        
        for name, eval in sorted(report.evaluations.items()):
            m = eval.metrics
            
            lines.append(f"### {name}")
            lines.append(f"")
            lines.append(f"**Overall Metrics:**")
            lines.append(f"- Accuracy: {m.accuracy:.2%} ({m.correct_samples}/{m.total_samples})")
            lines.append(f"- Macro F1: {m.macro_f1:.3f}")
            lines.append(f"- Macro Precision: {m.macro_precision:.3f}")
            lines.append(f"- Macro Recall: {m.macro_recall:.3f}")
            lines.append(f"")
            
            lines.append(f"**Cost Metrics:**")
            lines.append(f"- Total Cost: ${m.total_cost_usd:.4f}")
            lines.append(f"- Average Cost: ${m.avg_cost_usd:.5f}/query")
            lines.append(f"")
            
            lines.append(f"**Latency Metrics:**")
            lines.append(f"- Average: {m.avg_latency_ms:.0f}ms")
            lines.append(f"- P50: {m.latency_p50_ms:.0f}ms")
            lines.append(f"- P95: {m.latency_p95_ms:.0f}ms")
            lines.append(f"- P99: {m.latency_p99_ms:.0f}ms")
            lines.append(f"")
            
            lines.append(f"**Token Usage:**")
            lines.append(f"- Total: {m.total_tokens:,}")
            lines.append(f"- Average: {m.avg_tokens:.0f}/query")
            lines.append(f"")
            
            lines.append(f"**Per-Class Performance:**")
            lines.append(f"")
            lines.append(f"| Class | Precision | Recall | F1 | Support |")
            lines.append(f"|-------|-----------|--------|-----|---------|")
            
            for label, cm in m.class_metrics.items():
                lines.append(f"| {label} | {cm.precision:.3f} | {cm.recall:.3f} | {cm.f1:.3f} | {cm.support} |")
            
            lines.append(f"")
            
            # Confusion matrix
            if eval.confusion_matrix:
                lines.append(f"**Confusion Matrix:**")
                lines.append(f"")
                lines.append(f"| True \\ Pred | yes | no | maybe |")
                lines.append(f"|-------------|-----|-----|-------|")
                
                for gt_label in ["yes", "no", "maybe"]:
                    row = eval.confusion_matrix.get(gt_label, {})
                    lines.append(
                        f"| {gt_label} | {row.get('yes', 0)} | {row.get('no', 0)} | {row.get('maybe', 0)} |"
                    )
                
                lines.append(f"")
        
        # Cost-Efficiency Analysis
        lines.append(f"## Cost-Efficiency Analysis")
        lines.append(f"")
        
        # Calculate CER for each system
        lines.append(f"| System | CER (Cost/Accuracy) | Cost Savings vs Direct LLM |")
        lines.append(f"|--------|---------------------|---------------------------|")
        
        direct_llm_cost = None
        for name, eval in report.evaluations.items():
            if "direct" in name.lower():
                direct_llm_cost = eval.metrics.avg_cost_usd
                break
        
        for name, eval in sorted(report.evaluations.items()):
            m = eval.metrics
            cer = m.avg_cost_usd / m.accuracy if m.accuracy > 0 else float('inf')
            
            if direct_llm_cost and direct_llm_cost > 0:
                savings = (1 - m.avg_cost_usd / direct_llm_cost) * 100
                savings_str = f"{savings:+.1f}%"
            else:
                savings_str = "N/A"
            
            lines.append(f"| {name} | ${cer:.5f} | {savings_str} |")
        
        lines.append(f"")
        lines.append(f"*CER (Cost-Efficiency Ratio) = Average Cost / Accuracy. Lower is better.*")
        
        return "\n".join(lines)
    
    def save_markdown(self, report: EvaluationReport, filename: str = None) -> Path:
        """
        Save markdown report to file.
        
        Args:
            report: The evaluation report
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        markdown = self.generate_markdown(report)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        logger.info(f"Saved markdown report to {filepath}")
        
        return filepath
