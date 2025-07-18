import os
import pandas as pd
import numpy as np
from config import FP_DIR, LBL_DIR

def load_data(mech: int, fp_path: str, lbl_path: str = None):
    # Load labels
    if lbl_path and os.path.isfile(lbl_path):
        df_lbl = pd.read_csv(lbl_path)
    else:
        df_lbl = pd.read_csv(os.path.join(LBL_DIR, f'coche{mech}.csv'))

    smiles = df_lbl['smiles']
    y = df_lbl['label']

    # Load fingerprint
    df_fp = pd.read_csv(fp_path)
    X = df_fp.select_dtypes(include=[np.number])
    
    return X, y, smiles