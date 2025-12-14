import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# =========================================================
# 경로 설정
# =========================================================
ROOT = Path("/scratch/jjparkcv_root/jjparkcv98/minsukc/LIDC")
DICOM_ROOT = ROOT / "LIDC-IDRI_Data"     
NIFTI_ROOT = ROOT / "LIDC_NII_Data"   
NIFTI_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# DICOM series → NIfTI 변환 함수
# =========================================================
def dicom_series_to_nifti(dicom_dir: Path, out_path: Path):
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError("No DICOM series found")

    # LIDC는 보통 series 하나
    series_id = series_ids[0]
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)

    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    sitk.WriteImage(image, str(out_path))

# =========================================================
# 메인 변환 루프
# =========================================================
def main():
    patients = sorted([p for p in DICOM_ROOT.iterdir() if p.is_dir()])

    for idx, patient_dir in enumerate(
        tqdm(patients, desc="Converting DICOM → NIfTI")
    ):
        short_id = f"LIDC_{idx:04d}"
        out_file = NIFTI_ROOT / f"{short_id}.nii.gz"

        if out_file.exists():
            print("file exists. skipping...")
            continue

        try:
            dicom_series_to_nifti(patient_dir, out_file)
        except Exception as e:
            print(f"[ERROR] {patient_dir.name}: {e}")


if __name__ == "__main__":
    main()
