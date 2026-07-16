import kagglehub
import pandas as pd
from pathlib import Path

DATASET = "kuldeepjangra/e-commerce-customer-churn-dataset-200k-records"

def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download(DATASET)
    csv_files = list(Path(path).glob("*.csv"))
    print(f"발견된 파일: {[f.name for f in csv_files]}")
    return pd.read_csv(csv_files[0])

if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.columns.tolist())