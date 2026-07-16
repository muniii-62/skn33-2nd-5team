import kagglehub
import pandas as pd
from pathlib import Path

DATASET = "mashlyn/online-retail-ii-uci"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_CSV = DATA_DIR / "online_retail_II.csv"

def load_raw() -> pd.DataFrame:
    """Online Retail II 원본 로드. data/ 폴더에 없으면 kagglehub로 받아 저장."""
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV, encoding="ISO-8859-1")
    else:
        path = kagglehub.dataset_download(DATASET)
        csv = next(Path(path).glob("*.csv"))
        df = pd.read_csv(csv, encoding="ISO-8859-1")

        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(LOCAL_CSV, index=False)
        print(f"저장됨: {LOCAL_CSV}")

    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

if __name__ == "__main__":
    df = load_raw()
    print(df.shape)