 
from __future__ import annotations
 
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
 
import joblib
import numpy as np
import pandas as pd
 
from src.zeek_parser import ZeekParser
 
logger = logging.getLogger(__name__)
 
# Severity bands by attack probability
SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.95,
    "HIGH":     0.80,
    "MEDIUM":   0.60,
    "LOW":      0.40,
}
 
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS = {
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "HIGH":     Fore.RED,
        "MEDIUM":   Fore.YELLOW,
        "LOW":      Fore.CYAN,
        "RESET":    Style.RESET_ALL,
    }
except ImportError:
    COLORS = {k: "" for k in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "RESET")}
 
 def _severity(prob: float) -> str:
    for level, cutoff in SEVERITY_THRESHOLDS.items():
        if prob >= cutoff:
            return level
    return "INFO"
 
 
def _format_alert(alert: dict) -> str:
    sev = alert["severity"]
    color = COLORS.get(sev, "")
    reset = COLORS["RESET"]
    ts = alert.get("timestamp", "?")
    src = f"{alert.get('src_ip','?')}:{alert.get('src_port','?')}"
    dst = f"{alert.get('dst_ip','?')}:{alert.get('dst_port','?')}"
    prob = alert.get("attack_prob", 0.0)
    return (
        f"{color}[{sev}]{reset}  {ts}  "
        f"{src} → {dst}  "
        f"({alert.get('proto','?')})  "
        f"prob={prob:.3f}"
    )
 
 
class IDSDetector:
    """Load a model bundle and score Zeek connection records.
 
    Parameters
    ----------
    model_path : str | Path
        Path to the .pkl bundle saved by IDSTrainer.
    threshold : float
        Minimum attack probability to emit an alert.
    """
 
    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.70,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
 
        logger.info("Loading model from %s …", self.model_path)
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.feature_names: list[str] = bundle.get("feature_names", [])
        logger.info("Model ready — %d features", len(self.feature_names))
 
    
 
    def _align_features(self, df: pd.DataFrame) -> np.ndarray:
        """Align a DataFrame to the model's expected feature columns."""
        if not self.feature_names:
            raise RuntimeError("Model bundle is missing feature_names.")
 
        # Add missing columns as 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
 
        X = df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = X.values.astype(np.float32)
 
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
 
        return arr