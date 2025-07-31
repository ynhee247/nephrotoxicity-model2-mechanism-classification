import argparse
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint


parser = argparse.ArgumentParser(description="Generate RDKit fingerprints from a CSV file containing SMILES")
parser.add_argument('--input_file', required=True, help="Path to the input CSV file (file coche)")
args = parser.parse_args()

input_file = args.input_file
smiles_col = 'smiles'
output_dir = './data/fingerprints/'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
smiles_list = df[smiles_col].tolist()

# ==== Hàm fingerprint RDKit ====
def calc_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(MACCSkeys.GenMACCSKeys(mol)) if mol else np.nan

def calc_ecfp2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=2048)) if mol else np.nan

def calc_rdk7(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=4096)) if mol else np.nan

print("Tính MACCS (166 bit)...")
maccs_arr = np.array([calc_maccs(smi) for smi in smiles_list])
df_maccs = pd.DataFrame(maccs_arr)
df_maccs.insert(0, "smiles", smiles_list)
df_maccs.to_csv(os.path.join(output_dir, "maccs.csv"), index=False)
np.save(os.path.join(output_dir, "maccs.npy"), maccs_arr)

print("Tính ECFP2 (2048 bit)...")
ecfp2_arr = np.array([calc_ecfp2(smi) for smi in smiles_list])
df_ecfp2 = pd.DataFrame(ecfp2_arr)
df_ecfp2.insert(0, "smiles", smiles_list)
df_ecfp2.to_csv(os.path.join(output_dir, "ecfp2.csv"), index=False)
np.save(os.path.join(output_dir, "ecfp2.npy"), ecfp2_arr)

print("Tính RDK7 (4096 bit)...")
rdk7_arr = np.array([calc_rdk7(smi) for smi in smiles_list])
df_rdk7 = pd.DataFrame(rdk7_arr)
df_rdk7.insert(0, "smiles", smiles_list)
df_rdk7.to_csv(os.path.join(output_dir, "rdk7.csv"), index=False)
np.save(os.path.join(output_dir, "rdk7.npy"), rdk7_arr)