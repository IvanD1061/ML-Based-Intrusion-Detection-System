from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")   # headless rendering

logger = logging.getLogger(__name__)

# Consistent color palette
PALETTE = {
    "primary": "#2563EB",   # blue
    "danger": "#DC2626",    # red
    "success": "#16A34A",   # green
    "neutral": "#6B7280",   # grey
    "bg": "#F8FAFC",
}


class IDSEvaluator:
    """Load a saved model bundle and produce evaluation artefacts.

    Parameters
    ----------
    model_path : str | Path
        Path to the .pkl bundle saved by IDSTrainer.
    output_dir : str | Path
        Directory to write report figures.
    """

    def __init__(
        self,
        model_path: str | Path,
        output_dir: str | Path = "reports/",
    ) -> None:
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model bundle from %s …", self.model_path)
        self.bundle = joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.feature_names: list[str] = self.bundle.get("feature_names", [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 1. Classification Report
    # ------------------------------------------------------------------

    def print_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        labels = (
            ["BENIGN", "Attack"] if self.bundle.get("binary")
            else None
        )
        report = classification_report(y_true, y_pred, target_names=labels)
        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(report)
        return report

    # ------------------------------------------------------------------
    # 2. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Path:
        cm = confusion_matrix(y_true, y_pred)
        labels = ["BENIGN", "Attack"] if self.bundle.get("binary") else None

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(
            ax=ax,
            colorbar=False,
            cmap="Blues",
            values_format="d",
        )
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        return self._save(fig, "confusion_matrix.png")

    # ------------------------------------------------------------------
    # 3. ROC Curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Path:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2,
                label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1,
                linestyle="--", label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "roc_curve.png")

    # ------------------------------------------------------------------
    # 4. Precision-Recall Curve
    # ------------------------------------------------------------------

    def plot_pr_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Path:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        ax.plot(rec, prec, color=PALETTE["success"], lw=2)
        ax.fill_between(rec, prec, alpha=0.08, color=PALETTE["success"])

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "pr_curve.png")

    # ------------------------------------------------------------------
    # 5. Feature Importance
    # ------------------------------------------------------------------

    def plot_feature_importance(self, top_n: int = 20) -> Path:
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        names = (
            [self.feature_names[i] for i in indices]
            if self.feature_names
            else [f"feature_{i}" for i in indices]
        )
        values = importances[indices]

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        bars = ax.barh(
            range(top_n), values[::-1],
            color=PALETTE["primary"], alpha=0.85,
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")

        # Value labels
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
            )

        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "feature_importance.png")

    # ------------------------------------------------------------------
    # Main evaluation runner
    # ------------------------------------------------------------------

    def run(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Run all evaluations; returns a summary dict."""
        logger.info("Evaluating on %d samples …", len(y_test))

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        report = self.print_report(y_test, y_pred)

        paths = [
            self.plot_confusion_matrix(y_test, y_pred),
            self.plot_roc_curve(y_test, y_proba),
            self.plot_pr_curve(y_test, y_proba),
            self.plot_feature_importance(),
        ]
        logger.info("Reports written to %s", self.output_dir)

        auc = roc_auc_score(y_test, y_proba)
        return {
            "roc_auc": round(float(auc), 4),
            "n_test": len(y_test),
            "report_text": report,
            "plots": [str(p) for p in paths],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained ML-IDS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="models/rf_ids.pkl")
    p.add_argument("--data", default="data/cicids2017/")
    p.add_argument("--out", default="reports/")
    p.add_argument("--sample", type=float, default=0.2,
                   help="Fraction of data to use for evaluation")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    from src.feature_engineering import CICIDSPipeline

    pipeline = CICIDSPipeline(data_dir=args.data, sample_frac=args.sample)
    _, X_test, _, y_test = pipeline.run()

    evaluator = IDSEvaluator(model_path=args.model, output_dir=args.out)
    # Apply same scaler as training
    X_test_scaled = pipeline.scaler.transform(X_test)
    results = evaluator.run(X_test_scaled, y_test)
    print(f"\nROC-AUC: {results['roc_auc']}")


if __name__ == "__main__":
    main(sys.argv[1:])