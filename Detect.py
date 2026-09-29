 
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
 
 