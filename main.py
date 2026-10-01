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


def _cmd_train(args):
    from src.train import IDSTrainer
    trainer = IDSTrainer(
        data_dir=args.data,
        output_path=args.out,
        tune=args.tune,
        sample_frac=args.sample,
        cv_folds=args.folds,
    )
    trainer.run()


def _cmd_evaluate(args):
    from src.evaluate import IDSEvaluator
    from src.feature_engineering import CICIDSPipeline
    import joblib
    import numpy as np

    bundle = joblib.load(args.model)
    bundle_feature_names = bundle.get("feature_names", [])
    bundle_scaler = bundle.get("scaler")

    # Run pipeline without scaling so we can apply the saved scaler
    pipeline = CICIDSPipeline(data_dir=args.data, sample_frac=args.sample)
    _, X_test_scaled, _, y_test = pipeline.run()

    # Re-derive raw test data using the pipeline's inverse scaler path
    # Simpler: align to bundle features then apply bundle scaler
    import pandas as pd
    X_test_df = pd.DataFrame(X_test_scaled, columns=pipeline.feature_names)

    # Add any missing bundle features as zeros
    for col in bundle_feature_names:
        if col not in X_test_df.columns:
            X_test_df[col] = 0.0

    X_for_eval = X_test_df[bundle_feature_names].values.astype("float32") if bundle_feature_names else X_test_scaled

    # The pipeline already scaled with its own scaler; if bundle has a different
    # scaler we need raw data.  Since both are fitted on the same data distribution
    # (same CSV), use the bundle scaler on un-scaled values derived by inverse_transform.
    if bundle_scaler is not None and bundle_feature_names:
        # Inverse-transform to get raw, then re-scale with bundle scaler
        try:
            X_raw = pipeline.scaler.inverse_transform(
                X_test_df[pipeline.feature_names].values.astype("float32")
            )
            X_raw_df = pd.DataFrame(X_raw, columns=pipeline.feature_names)
            for col in bundle_feature_names:
                if col not in X_raw_df.columns:
                    X_raw_df[col] = 0.0
            X_for_eval = bundle_scaler.transform(
                X_raw_df[bundle_feature_names].values.astype("float32")
            )
        except Exception:
            pass  # fall back to already-scaled data

    evaluator = IDSEvaluator(model_path=args.model, output_dir=args.out)
    results = evaluator.run(X_for_eval, y_test)
    print(f"\nROC-AUC: {results['roc_auc']}")
    print(f"Plots saved to: {args.out}")


def _cmd_detect(args):
    from src.detect import main as _main
    # Pass remaining args through to detect's own parser
    sys.argv = ["detect"] + sys.argv[2:]
    _main()


def main():
    p = argparse.ArgumentParser(
        description="ML-Based Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate synthetic sample dataset")
    gen.add_argument("--out", default="data/cicids2017/sample.csv")
    gen.add_argument("--rows", type=int, default=5000)
    gen.add_argument("--attack-ratio", type=float, default=0.30)

    # --- train ---
    tr = sub.add_parser("train", help="Train the Random Forest model")
    tr.add_argument("--data", default="data/cicids2017/",
                    help="CICIDS2017 CSV directory")
    tr.add_argument("--out", default="models/rf_ids.pkl",
                    help="Output model bundle path")
    tr.add_argument("--tune", action="store_true", help="GridSearchCV tuning")
    tr.add_argument("--sample", type=float, default=1.0,
                    help="Fraction of data to use (0.1 = 10%% for quick test)")
    tr.add_argument("--folds", type=int, default=5)

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Evaluate a trained model")
    ev.add_argument("--model", default="models/rf_ids.pkl")
    ev.add_argument("--data", default="data/cicids2017/")
    ev.add_argument("--out", default="reports/")
    ev.add_argument("--sample", type=float, default=0.2)

    # --- detect ---
    sub.add_parser("detect", help="Run detection (batch or watch)",
                   add_help=False)

    args, _ = p.parse_known_args()

    dispatch = {
        "generate": _cmd_generate,
        "train":    _cmd_train,
        "evaluate": _cmd_evaluate,
        "detect":   _cmd_detect,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()