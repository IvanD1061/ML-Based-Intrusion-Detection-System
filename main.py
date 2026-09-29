 
import argparse
import logging
import sys
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
 
 
def _cmd_generate(args):
    from src.generate_sample import main as _main
    argv = ["--out", args.out, "--rows", str(args.rows),
            "--attack-ratio", str(args.attack_ratio)]
    _main(argv)
 