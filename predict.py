import argparse
import pandas as pd
from utils import load_model
import logging

def load_fingerprints(path: str):
    """Return metadata columns and fingerprint matrix."""
    df = pd.read_csv(path)
    meta = df[['Name', 'smiles']]
    feature_cols = [c for c in df.columns if c not in meta.columns]
    X = df[feature_cols]
    return meta, X


def main():
    parser = argparse.ArgumentParser(description='Predict nephrotoxicity mechanism')
    parser.add_argument('--model', required=True, help='Path to trained .joblib model')
    parser.add_argument('--input', required=True, help='CSV file with Name, smiles and fingerprint columns')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    meta, X = load_fingerprints(args.input)

    model = load_model(args.model)

    preds = model.predict(X)
    result = meta.copy()
    result['prediction'] = preds

    if hasattr(model, 'predict_proba'):
        result['probability'] = model.predict_proba(X)[:, 1]

    result.to_csv(args.output, index=False)
    print(f'Saved predictions to {args.output}')


if __name__ == "__main__":
    main()