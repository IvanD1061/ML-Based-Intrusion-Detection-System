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
 
    # ------------------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------------------
 
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
 
    # ------------------------------------------------------------------
    # Score a DataFrame of connections
    # ------------------------------------------------------------------
 
    def score(self, conn_df: pd.DataFrame) -> pd.DataFrame:
        """Return conn_df with added columns: attack_prob, severity, alert."""
        if conn_df is None or conn_df.empty:
            return pd.DataFrame()
 
        X = self._align_features(conn_df.copy())
        proba = self.model.predict_proba(X)[:, 1]
 
        result = conn_df.copy()
        result["attack_prob"] = proba
        result["severity"] = [_severity(p) for p in proba]
        result["alert"] = proba >= self.threshold
        return result
 
    # ------------------------------------------------------------------
    # Build alert dicts
    # ------------------------------------------------------------------
 
    def to_alerts(self, scored_df: pd.DataFrame) -> list[dict]:
        """Convert a scored DataFrame to a list of alert records."""
        alerts = scored_df[scored_df["alert"]].copy()
        if alerts.empty:
            return []
 
        records = []
        for _, row in alerts.iterrows():
            ts_raw = row.get("ts", None)
            if ts_raw is not None and not pd.isna(ts_raw):
                try:
                    ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).isoformat()
                except (ValueError, OSError):
                    ts = str(ts_raw)
            else:
                ts = datetime.now(tz=timezone.utc).isoformat()
 
            record = {
                "timestamp": ts,
                "uid": str(row.get("uid", "")),
                "src_ip": str(row.get("id.orig_h", "")),
                "src_port": int(row.get("id.orig_p", 0)),
                "dst_ip": str(row.get("id.resp_h", "")),
                "dst_port": int(row.get("id.resp_p", 0)),
                "proto": str(row.get("proto", "")),
                "service": str(row.get("service", "")),
                "duration": float(row.get("duration", 0.0)),
                "orig_bytes": float(row.get("orig_bytes", 0.0)),
                "resp_bytes": float(row.get("resp_bytes", 0.0)),
                "attack_prob": round(float(row["attack_prob"]), 4),
                "severity": str(row["severity"]),
                "alert": True,
            }
            records.append(record)
        return records
 
    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
 
    def run_batch(
        self,
        log_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """Process all logs in log_dir and return (and optionally write) alerts."""
        parser = ZeekParser(log_dir)
        conn_df = parser.parse_conn_log()
        if conn_df is None or conn_df.empty:
            logger.warning("No conn.log data found in %s", log_dir)
            return []
 
        logger.info("Scoring %d connections …", len(conn_df))
        scored = self.score(conn_df)
        alerts = self.to_alerts(scored)
 
        n_total = len(scored)
        n_alerts = len(alerts)
        logger.info("Alerts: %d / %d  (%.1f%%)", n_alerts, n_total,
                    100 * n_alerts / max(n_total, 1))
 
        for a in alerts:
            print(_format_alert(a))
 
        if output_path and alerts:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as fh:
                for a in alerts:
                    fh.write(json.dumps(a) + "\n")
            logger.info("Alerts written → %s", out)
 
        return alerts
 
    # ------------------------------------------------------------------
    # Watch mode (tail live logs)
    # ------------------------------------------------------------------
 
    def run_watch(
        self,
        log_dir: str | Path,
        poll_interval: float = 5.0,
    ) -> None:
        """Poll log_dir every poll_interval seconds for new connections."""
        log_dir = Path(log_dir)
        seen_uids: set[str] = set()
 
        logger.info(
            "Watching %s  (poll every %.0fs) — Ctrl+C to stop",
            log_dir, poll_interval,
        )
 
        while True:
            try:
                parser = ZeekParser(log_dir)
                conn_df = parser.parse_conn_log()
                if conn_df is not None and not conn_df.empty:
                    # Only process new UIDs
                    if "uid" in conn_df.columns:
                        new = conn_df[~conn_df["uid"].isin(seen_uids)]
                        seen_uids.update(conn_df["uid"].tolist())
                    else:
                        new = conn_df
 
                    if not new.empty:
                        scored = self.score(new)
                        alerts = self.to_alerts(scored)
                        for a in alerts:
                            print(_format_alert(a))
 
                time.sleep(poll_interval)
 
            except KeyboardInterrupt:
                logger.info("Watch stopped.")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error during watch: %s", exc)
                time.sleep(poll_interval)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ML-IDS detection engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)
 
    # Batch
    batch_p = sub.add_parser("batch", help="Process a directory of Zeek logs")
    batch_p.add_argument("--logs", required=True, help="Zeek log directory")
    batch_p.add_argument("--model", default="models/rf_ids.pkl")
    batch_p.add_argument("--threshold", type=float, default=0.70)
    batch_p.add_argument("--out", default=None, help="Output .jsonl alert file")
 
    # Watch
    watch_p = sub.add_parser("watch", help="Tail a live Zeek log directory")
    watch_p.add_argument("--logs", required=True, help="Zeek log directory")
    watch_p.add_argument("--model", default="models/rf_ids.pkl")
    watch_p.add_argument("--threshold", type=float, default=0.70)
    watch_p.add_argument("--interval", type=float, default=5.0)
 
    return p
 
 
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_arg_parser().parse_args(argv)
    detector = IDSDetector(model_path=args.model, threshold=args.threshold)
 
    if args.mode == "batch":
        detector.run_batch(log_dir=args.logs, output_path=args.out)
    elif args.mode == "watch":
        detector.run_watch(log_dir=args.logs, poll_interval=args.interval)
 
 
if __name__ == "__main__":
    main(sys.argv[1:])