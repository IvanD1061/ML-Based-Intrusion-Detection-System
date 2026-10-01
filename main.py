 
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