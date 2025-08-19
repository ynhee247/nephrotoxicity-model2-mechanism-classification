import os
import argparse
import pandas as pd
import json

from config import MODELS, RANDOM_STATE, FP_DIR, LBL_DIR, DEVICE, CV_SPLITTER
from data_loader import load_data
from preprocess import resample_data
from model_training import train_model, PARAM_GRIDS, build_pipeline
from evaluation import evaluate_model, plot_confusion_matrix, oof_eval_for_params
from utils import save_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
    parser.add_argument('--oof_cm', action='store_true',
                        help='If set, compute and attach OOF confusion matrix per parameter set into *_all_cv.csv')
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
                sampler = SMOTE(random_state=RANDOM_STATE) if method == 'smote' else RandomUnderSampler(random_state=RANDOM_STATE)

                for model_name in MODELS:
                    print(f"Training {model_name.upper()} | FP={fp_name} | Method={method.upper()}")
                    
                    # Run GRID SEARCH without refitting to allow manual selection later (trên TRAIN gốc; sampler nằm trong pipeline → tránh leakage)
                    _, _, cv_results = train_model(
                        X_train, y_train,
                        model_name,
                        refit_metric=None,
                        params=None,
                        sampler=sampler
                    )

                    df_cv = pd.DataFrame(cv_results)

                    # (OPTIONAL) Tính & gắn OOF-CM nếu flag bật
                    if args.oof_cm:
                        pipe = build_pipeline(model_name, sampler)
                        df_cv['oof_confusion_matrix_cv'] = None
                        df_cv['oof_accuracy_cv'] = None
                        df_cv['oof_precision_cv'] = None
                        df_cv['oof_recall_cv'] = None
                        df_cv['oof_f1_cv'] = None
                        df_cv['oof_roc_auc_cv'] = None

                        print(f"Computing OOF-CM for {len(df_cv)} parameter sets...")
                        for i, p in enumerate(df_cv['params']):
                            oof = oof_eval_for_params(pipe, X_train, y_train, p, CV_SPLITTER)
                            cm_json = json.dumps({'labels': [0, 1], 'matrix': oof['cm'].tolist()})
                            df_cv.at[i, 'oof_confusion_matrix_cv'] = cm_json
                            df_cv.at[i, 'oof_accuracy_cv']        = oof['accuracy']
                            df_cv.at[i, 'oof_precision_cv']       = oof['precision']
                            df_cv.at[i, 'oof_recall_cv']          = oof['recall']
                            df_cv.at[i, 'oof_f1_cv']              = oof['f1']
                            df_cv.at[i, 'oof_roc_auc_cv']         = oof['roc_auc']
                    
                    # Metadata
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
                mech_sel = int(row['mechanism'])
                model_name = row['model']
                resample_method = row['resample']
                fp_name_sel = row['fingerprint']

                print(f"\nRetraining best model for mechanism {mech} - {model_name}_{resample_method}_{fp_name_sel}")

                # Map params từ file best_config sang clf__*
                params = {}
                for k in row.index:
                    if k in ['mechanism', 'model', 'resample', 'fingerprint'] or pd.isna(row[k]):
                        continue
                    params[k if str(k).startswith('clf__') else f'clf__{k}'] = row[k]

                # Retrain best model on full train set
                lbl_path = get_lbl_path(args.lbl_dir, mech)
                fp_path = next(p for p in fp_paths if os.path.splitext(os.path.basename(p))[0] == fp_name_sel)
                X_all, y_all, _ = load_data(mech, fp_path, lbl_path)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.3, stratify=y_all, random_state=RANDOM_STATE
                )

                sampler = SMOTE(random_state=RANDOM_STATE) if resample_method == 'smote' else RandomUnderSampler(random_state=RANDOM_STATE)

                final_model, _, _ = train_model(
                    X_train, y_train, model_name,
                    params=params, sampler=sampler
                )

                # Evaluate best model on test set
                final_metrics = evaluate_model(final_model, X_test, y_test)
                print(f"Test ROC AUC for mechanism {mech}: {final_metrics['roc_auc']:.4f}")
                print(final_metrics['report'])

                # Plot and save confusion matrix
                cm_file = os.path.join(
                    out_dir,
                    f'confusion_coche{mech}_{model_name}_{resample_method}_{fp_name_sel}.png'
                )
                plot_confusion_matrix(
                    final_metrics['confusion_matrix'],
                    labels=['Non-toxic', 'Toxic'],
                    filename=cm_file,
                    title=f'Ma trận nhầm lẫn - cơ chế {mech}',
                    xlabel='Dự đoán',
                    ylabel='Thực tế'
                )
                print(f"Saved confusion matrix to {cm_file}")

                # Save the best model
                best_model = f"best_coche{mech}_{model_name}_{resample_method}_{fp_name_sel}.joblib"
                save_model(final_model, os.path.join(out_dir, best_model))
                print(f"Saved best model for mechanism {mech} to {best_model}")

                report = final_metrics['report']
                summary.append({
                    'mechanism': mech,
                    'model': model_name,
                    'resample': resample_method,
                    'fingerprint': fp_name_sel,
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