from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)

# Realistic feature ranges derived from CICIDS2017 statistics
FEATURE_SPEC = {
    "Flow Duration":             (0,    120_000_000),
    "Total Fwd Packets":         (1,    10_000),
    "Total Backward Packets":    (0,    10_000),
    "Total Length of Fwd Packets": (0,  100_000),
    "Total Length of Bwd Packets": (0,  100_000),
    "Fwd Packet Length Max":     (0,    65535),
    "Fwd Packet Length Min":     (0,    1500),
    "Fwd Packet Length Mean":    (0,    1500),
    "Fwd Packet Length Std":     (0,    1500),
    "Bwd Packet Length Max":     (0,    65535),
    "Bwd Packet Length Min":     (0,    1500),
    "Bwd Packet Length Mean":    (0,    1500),
    "Bwd Packet Length Std":     (0,    1500),
    "Flow Bytes/s":              (0,    1_000_000),
    "Flow Packets/s":            (0,    100_000),
    "Flow IAT Mean":             (0,    10_000_000),
    "Flow IAT Std":              (0,    10_000_000),
    "Flow IAT Max":              (0,    120_000_000),
    "Flow IAT Min":              (0,    120_000_000),
    "Fwd IAT Total":             (0,    120_000_000),
    "Fwd IAT Mean":              (0,    10_000_000),
    "Fwd IAT Std":               (0,    10_000_000),
    "Fwd IAT Max":               (0,    120_000_000),
    "Fwd IAT Min":               (0,    120_000_000),
    "Bwd IAT Total":             (0,    120_000_000),
    "Bwd IAT Mean":              (0,    10_000_000),
    "Bwd IAT Std":               (0,    10_000_000),
    "Bwd IAT Max":               (0,    120_000_000),
    "Bwd IAT Min":               (0,    120_000_000),
    "Fwd PSH Flags":             (0,    1),
    "Bwd PSH Flags":             (0,    1),
    "Fwd URG Flags":             (0,    1),
    "Bwd URG Flags":             (0,    1),
    "Fwd Header Length":         (0,    65535),
    "Bwd Header Length":         (0,    65535),
    "Fwd Packets/s":             (0,    100_000),
    "Bwd Packets/s":             (0,    100_000),
    "Min Packet Length":         (0,    1500),
    "Max Packet Length":         (0,    65535),
    "Packet Length Mean":        (0,    1500),
    "Packet Length Std":         (0,    1500),
    "Packet Length Variance":    (0,    2_250_000),
    "FIN Flag Count":            (0,    1),
    "SYN Flag Count":            (0,    1),
    "RST Flag Count":            (0,    1),
    "PSH Flag Count":            (0,    1),
    "ACK Flag Count":            (0,    1),
    "URG Flag Count":            (0,    1),
    "CWE Flag Count":            (0,    1),
    "ECE Flag Count":            (0,    1),
    "Down/Up Ratio":             (0,    10),
    "Average Packet Size":       (0,    1500),
    "Avg Fwd Segment Size":      (0,    1500),
    "Avg Bwd Segment Size":      (0,    1500),
    "Fwd Header Length.1":       (0,    65535),
    "Subflow Fwd Packets":       (1,    10_000),
    "Subflow Fwd Bytes":         (0,    100_000),
    "Subflow Bwd Packets":       (0,    10_000),
    "Subflow Bwd Bytes":         (0,    100_000),
    "Init_Win_bytes_forward":    (-1,   65535),
    "Init_Win_bytes_backward":   (-1,   65535),
    "act_data_pkt_fwd":          (0,    10_000),
    "min_seg_size_forward":      (0,    1500),
    "Active Mean":               (0,    10_000_000),
    "Active Std":                (0,    10_000_000),
    "Active Max":                (0,    120_000_000),
    "Active Min":                (0,    120_000_000),
    "Idle Mean":                 (0,    120_000_000),
    "Idle Std":                  (0,    120_000_000),
    "Idle Max":                  (0,    120_000_000),
    "Idle Min":                  (0,    120_000_000),
}

ATTACK_LABELS = [
    "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
    "DDoS", "PortScan", "FTP-Patator", "SSH-Patator",
    "Web Attack – Brute Force", "Bot", "Infiltration",
]

def _make_rows(n: int, label: str, scale: float = 1.0) -> pd.DataFrame:
    data = {}
    for col, (lo, hi) in FEATURE_SPEC.items():
        if hi == 1:  # binary flag
            data[col] = RNG.integers(0, 2, size=n).astype(float)
        else:
            vals = RNG.uniform(lo * scale, hi * scale, size=n)
            data[col] = np.clip(vals, lo, hi)
    data["Label"] = label
    return pd.DataFrame(data)

def generate(n_rows: int = 5000, attack_ratio: float = 0.3) -> pd.DataFrame:
    n_attack = int(n_rows * attack_ratio)
    n_benign = n_rows - n_attack
 
    frames = [_make_rows(n_benign, "BENIGN")]
 
    # Distribute attack rows across attack types
    labels = RNG.choice(ATTACK_LABELS, size=n_attack)
    for label in ATTACK_LABELS:
        count = int((labels == label).sum())
        if count > 0:
            # Attacks have different traffic patterns
            frames.append(_make_rows(count, label, scale=1.5))
 
    df = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
 
    logger.info(
        "Generated %d rows  (BENIGN=%d, Attack=%d)",
        len(df), n_benign, n_attack,
    )
    return df
 
 def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    p = argparse.ArgumentParser(description="Generate synthetic CICIDS2017-like sample")
    p.add_argument("--out", default="data/cicids2017/sample.csv")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--attack-ratio", type=float, default=0.30)
    args = p.parse_args(argv)
 
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_rows=args.rows, attack_ratio=args.attack_ratio)
    df.to_csv(out, index=False)
    logger.info("Saved → %s", out)
 


if __name__ == "__main__":
    main()