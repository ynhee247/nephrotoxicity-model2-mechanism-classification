# MODEL PHÂN LOẠI CƠ CHẾ GÂY ĐỘC THẬN

## Fingerprints
Thư viện sử dụng:
- RDKit: MACCs (166 bit), ECFP2 (2048 bit), RDK7 (4096 bit)
- Padelpy: PubChem (881 bit), Klekota-Roth Count (4860 bit), Substructure Count (307 bit)

==> 6 bộ fingerprints

*Xử lý fingerprint RDKIT:*
- Run `python fingerprint_rdkit.py --input_file <path>` => 3 files fingerprint RDKit: maccs, ecfp2, rdk7 (.csv, .npy)

*Xử lý fingerprint Padelpy*:
- Tạo file smiles.smi (data/fingerprints)
- **PaDEL-Descriptor/descriptors.xml** => `<Group name="Fingerprint">` => Chọn fingerprint mình muốn rồi chỉnh `value="true"`, mấy fingerprint khác thì `value="false"` (mặc định đang là Pubchem)
- Ví dụ:
+ `<Descriptor name="PubchemFingerprinter" value="true"/>`
+ `<Descriptor name="SubstructureFingerprintCount" value="true"/>`
+ `<Descriptor name="KlekotaRothFingerprintCount" value="true"/>`
- Run `java -jar <path to PaDEL-Descriptor.jar> -fingerprints -dir <path to smiles.smi> -file <path to output file>` (output file: data/fingerprints, tạo sẵn file .csv rồi copy path vô)
- Sort lại name (A to Z) để đúng thứ tự smiles so với file coche => xóa cột **Name** luôn

==> Folder **data/** gồm 2 folder **labels (chứa 7 files 7 cơ chế: coche1.csv...)** + **fingerprints (chứa 6 files .csv fingerprint, xóa mấy file thừa đi)**

## Training
- Chỉ run file **main.py**
- Run 7 lần cho 7 cơ chế
- Run `python main.py --mechanism <chọn 1 cơ chế, số 1 -> 7)> --fp_dir data/fingerprints --lbl_dir <path to file .csv coche> --out_dir <path to output folder>` (output folder: tạo folder **results**)

- Folder output sẽ chứa:
+ File .csv chứa kết quả training của từng cơ chế: vd **coche1_all_cv.csv**...
+ Best model cho từng cơ chế (.joblib): vd **best_coche1_xgb_smote_pubchem.joblib**...

## Predicting
- Chuẩn bị file input đã **precompute fingerprint** gồm các cột: Name, smiles, fp...
- Run `python predict.py --model <path (.joblib)> --input <path>`