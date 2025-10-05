# MODEL PHÂN LOẠI CƠ CHẾ GÂY ĐỘC THẬN


## Fingerprints
Thư viện sử dụng:
- RDKit: MACCs (166 bit), ECFP2 (2048 bit), RDK7 (4096 bit)
- Padelpy: PubChem (881 bit), Klekota-Roth Count (4860 bit), Substructure Count (307 bit)

==> 6 bộ fingerprints

### Xử lý fingerprint RDKIT ###
Run `python fingerprint_rdkit.py --input_file <path>`
=> 6 files fingerprint RDKit: maccs, ecfp2, rdk7 (.csv, .npy)

### Xử lý fingerprint Padelpy ###
Tạo file smiles.smi (data/fingerprints)
**PaDEL-Descriptor/descriptors.xml** => `<Group name="Fingerprint">` => Chọn fingerprint mình muốn rồi chỉnh `value="true"`, mấy fingerprint khác thì `value="false"` (mặc định đang là Pubchem)
Ví dụ:
+ `<Descriptor name="PubchemFingerprinter" value="true"/>`
+ `<Descriptor name="SubstructureFingerprintCount" value="true"/>`
+ `<Descriptor name="KlekotaRothFingerprintCount" value="true"/>`
Run `java -jar <path to PaDEL-Descriptor.jar> -fingerprints -dir <path to smiles.smi> -file <path>` (output file: data/fingerprints)
Sort lại name (A to Z) để đúng thứ tự smiles so với file coche => xóa cột **Name** luôn

==> Folder **data/** gồm 2 folder **labels (chứa 7 files 7 cơ chế: coche1.csv...)** + **fingerprints (chứa 6 files .csv fingerprint: maccs.csv...)**

## Training
Run 7 lần cho 7 cơ chế
Run `python main.py --mechanism <chọn 1 cơ chế, số 1 -> 7)> --fp_dir data/fingerprints --lbl_dir <path to file .csv coche> --out_dir <path to output folder>` (output folder: tạo folder **results**)

Folder **results** sẽ chứa: 
+ File .csv chứa kết quả training của từng cơ chế: vd **coche1_all_cv.csv**... => tự đánh giá và chọn bộ best params
+ Tạo file **best_config.csv (./results)** => mỗi dòng chứa thông tin cấu hình của best model cho từng cơ chế


## Retrain best model
Run `python main.py --fp_dir <data/fingerprints> --lbl_dir <data/labels> --out_dir <results/retrain-and-evaluate> --best_config <path to best_config.csv> --skip_cv`


## Predicting
Chuẩn bị file input đã **precompute fingerprint** gồm các cột: Name, smiles, fp...
Run `python predict.py --model <path (.joblib)> --input <path> --output <path>`