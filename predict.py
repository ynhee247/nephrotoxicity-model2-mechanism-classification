import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from utils import load_model
import logging

logger = logging.getLogger(__name__)


def infer_fp_type(model_path: str) -> str:
    """Guess fingerprint type from model filename."""
    name = os.path.splitext(os.path.basename(model_path))[0]
    parts = name.split('_')
    return parts[-1] if len(parts) >= 5 else 'ecfp2'


def calc_maccs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 166
    return list(MACCSkeys.GenMACCSKeys(mol))


def calc_ecfp2(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 2048
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=2048)
    return list(fp)


def calc_rdk7(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 4096
    fp = RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=4096)
    return list(fp)


def _compute_padel_fp(smiles: str, prefix: str, length: int):
    """Helper to compute PadelPy based fingerprints."""
    try:
        from padelpy import from_smiles
    except ImportError as exc:
        raise ImportError('padelpy is required for this fingerprint') from exc

    try:
        df = from_smiles(smiles, fingerprints=True, descriptors=False)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([df])
    except Exception as exc:
        logger.warning('Failed to compute %s fingerprint: %s', prefix, exc)
        return [0] * length

    cols = [c for c in df.columns if c.startswith(prefix)]
    return df.loc[0, cols].astype(int).tolist()


def calc_pubchem(smiles: str):
    return _compute_padel_fp(smiles, 'PubchemFP', 881)


def calc_klekota_roth_count(smiles: str):
    return _compute_padel_fp(smiles, 'KlekotaRothFP', 4860)


def calc_substructure_count(smiles: str):
    return _compute_padel_fp(smiles, 'SubstructureFingerprintCount', 307)


def smiles_to_fingerprint(smiles_series, fp_type: str):
    calc_fn = {
        'maccs': calc_maccs,
        'ecfp2': calc_ecfp2,
        'rdk7': calc_rdk7,
        'pubchem': calc_pubchem,
        'klekota': calc_klekota_roth_count,
        'substructure': calc_substructure_count,
    }.get(fp_type.lower())
    if calc_fn is None:
        raise ValueError(f'Unsupported fingerprint type: {fp_type}')
    arr = [calc_fn(smi) for smi in smiles_series]
    return pd.DataFrame(arr)


def main():
    parser = argparse.ArgumentParser(description='Predict nephrotoxicity mechanism')
    parser.add_argument('--model', required=True, help='Path to trained .joblib model')
    parser.add_argument('--input', required=True, help='CSV with a "smiles" column')
    parser.add_argument(
        '--fp_type',
        help='Fingerprint type (maccs, ecfp2, rdk7, pubchem, klekota, substructure). '
             'If omitted, infer from model name'
    )
    parser.add_argument('--output', default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()

    fp_type = args.fp_type or infer_fp_type(args.model)
    df = pd.read_csv(args.input)
    if 'smiles' not in df.columns:
        raise ValueError('Input file must contain a "smiles" column')

    X = smiles_to_fingerprint(df['smiles'], fp_type)
    model = load_model(args.model)

    preds = model.predict(X)
    result = pd.DataFrame({'smiles': df['smiles'], 'prediction': preds})
    if hasattr(model, 'predict_proba'):
        result['probability'] = model.predict_proba(X)[:, 1]

    result.to_csv(args.output, index=False)
    print(f'Saved predictions to {args.output}')


if __name__ == "__main__":
    main()