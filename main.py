import os
import argparse
import pandas as pd
from config import MODELS, RANDOM_STATE, FP_DIR, LBL_DIR, DEVICE
from data_loader import load_data
from preprocess import resample_data
from model_training import train_model
from evaluation import evaluate_model, plot_confusion_matrix
from utils import save_model
from sklearn.model_selection import train_test_split

def get_fp_paths(fp_arg: str):
    if fp_arg and os.path.isfile(fp_arg):
        return [os.path.abspath(fp_arg)]
    dirp = fp_arg or FP_DIR
    return [os.path.join(dirp, f)
            for f in sorted(os.listdir(dirp))
            if f.lower().endswith('.csv')]

def get_lbl_path(lbl_arg: str, mech: int):
    if lbl_arg and os.path.isfile(lbl_arg):
        return os.path.abspath(lbl_arg)
    return os.path.join(LBL_DIR, f'coche{mech}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run renal toxicity classification')
    parser.add_argument('--mechanism', type=int,
                        help='Mechanism number (1-7). If omitted, runs all mechanisms')
    parser.add_argument('--fp_dir', type=str,
                        help='Path to fingerprint CSV file or directory of CSVs')
    parser.add_argument('--lbl_dir', type=str,
                        help='Path to label CSV file or directory of label CSVs')
    parser.add_argument('--out_dir', type=str, default='models',
                        help='Directory to save trained models and CV results')
    parser.add_argument('--best_config', type=str,
                        help='CSV file specifying selected model/resample/fingerprint for each mechanism')
    args = parser.parse_args()

    # Display which device will be used for training
    print(f"Device: {DEVICE}")
    
    # Prepare output directory
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Prepare fingerprint file list and mechanisms
    fp_paths = get_fp_paths(args.fp_dir)
    mechanisms = [args.mechanism] if args.mechanism else list(range(1, 8))

    # Run cross-validation
    for mech in mechanisms:
        lbl_path = get_lbl_path(args.lbl_dir, mech)
        mech_cv = []

        for fp_path in fp_paths:
            fp_name = os.path.splitext(os.path.basename(fp_path))[0]
            print(f"\n--- Cơ chế {mech} - Fingerprint {fp_name} ---")

            # Load data
            X, y, smiles = load_data(mech, fp_path, lbl_path)

            print(f"Full dataset: total={len(y)} samples => 0: {sum(y==0)}, 1: {sum(y==1)}")

            # Export fingerprint matrix
            out_fp_csv = f"fingerprint_matrix_coche{mech}_{fp_name}.csv"
            X.to_csv(os.path.join(out_dir, out_fp_csv), index=False)
            print(f"Saved matrix to {out_fp_csv}")

            # Stratified train/test split (train:test = 7:3)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3,
                stratify=y, random_state=RANDOM_STATE
            )

            for method in ['smote', 'rus']:
                X_res, y_res = resample_data(X_train, y_train, method)
                for model_name in MODELS:
                    print(f"Training {model_name.upper()} | FP={fp_name} | Method={method.upper()} | refit=False")
                    _, _, cv_results = train_model(X_res, y_res, model_name, refit_metric=None)

                    # Save individual CV results
                    df_cv = pd.DataFrame(cv_results)
                    df_cv['mechanism']   = mech
                    df_cv['model']       = model_name
                    df_cv['resample']    = method
                    df_cv['fingerprint'] = fp_name
                    mech_cv.append(df_cv)
        
        mech_df = pd.concat(mech_cv, ignore_index=True)
        mech_cv_file = os.path.join(out_dir, f'coche{mech}_all_cv.csv')
        mech_df.to_csv(mech_cv_file, index=False)
        print(f"Saved combined CV for mechanism {mech} to {mech_cv_file}")

        # Retrain/evaluate selected models
        if args.best_config:
            best_cfg = pd.read_csv(args.best_config)
            summary = []
            for _, row in best_cfg.iterrows():
                mech = int(row['mechanism'])
                model_name = row['model']
                resample_method = row['resample']
                fp_name = row['fingerprint']

                print(f"\nRetraining best model for mechanism {mech} - {model_name}_{resample_method}_{fp_name}")

                # Retrain best model on full train set
                lbl_path = get_lbl_path(args.lbl_dir, mech)
                fp_path = next(p for p in fp_paths if os.path.splitext(os.path.basename(p))[0]==fp_name)

                X, y, _ = load_data(mech, fp_path, lbl_path)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3,
                    stratify=y, random_state=RANDOM_STATE
                )
                X_res, y_res = resample_data(X_train, y_train, resample_method)
                final_model, _, _ = train_model(X_res, y_res, model_name)

                # Evaluate best model on test set
                final_metrics = evaluate_model(final_model, X_test, y_test)
                print(f"Test ROC AUC for mechanism {mech}: {final_metrics['roc_auc']:.4f}")
                print(final_metrics['report'])

                # Plot and save confusion matrix
                cm_file = os.path.join(
                    out_dir,
                    f'confusion_coche{mech}_{model_name}_{resample_method}_{fp_name}.png'
                )
                plot_confusion_matrix(
                    final_metrics['confusion_matrix'],
                    labels=['Non-toxic', 'Toxic'],
                    filename=cm_file
                )
                print(f"Saved confusion matrix to {cm_file}")

                # Save the best model
                best_model = f"best_coche{mech}_{model_name}_{resample_method}_{fp_name}.joblib"
                save_model(final_model, os.path.join(out_dir, best_model))
                print(f"Saved best model for mechanism {mech} to {best_model}")

                report = final_metrics['report']
                summary.append({
                    'mechanism': mech,
                    'model': model_name,
                    'resample': resample_method,
                    'fingerprint': fp_name,
                    'accuracy': report['accuracy'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'roc_auc': final_metrics['roc_auc']
                })

            summary_df = pd.DataFrame(summary)
            summary_file = os.path.join(out_dir, 'best_per_mechanism.csv')
            # If the summary CSV exists, append without header; otherwise write new
            if os.path.exists(summary_file):
                summary_df.to_csv(summary_file, index=False, mode='a', header=False)
            else:
                summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary of best models to {summary_file}")
        print("\nDone.")