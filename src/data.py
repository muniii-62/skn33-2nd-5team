import kagglehub
import pandas as pd
from pathlib import Path

DATASET = "mashlyn/online-retail-ii-uci"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LOCAL_CSV = DATA_DIR / "online_retail_II.csv"

def load_raw() -> pd.DataFrame:
    """Online Retail II 원본 로드. data/ 폴더에 없으면 kagglehub로 받아 저장."""
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV, encoding="ISO-8859-1")
    else:
        path = kagglehub.dataset_download(DATASET)
        csv = next(Path(path).glob("*.csv"))
        df = pd.read_csv(csv, encoding="ISO-8859-1")

        DATA_DIR.mkdir(parents=True,exist_ok=True)
        df.to_csv(LOCAL_CSV, index=False)
        print(f"저장됨: {LOCAL_CSV}")

    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

def load_sales() -> pd.DataFrame:
    """정제된 판매 데이터: 취소·비상품·내부기록·비회원 제외한 회원의 정상 구매만"""
    df = load_raw()
    is_cancel  = df["Invoice"].astype(str).str.startswith("C")
    is_product = df["StockCode"].astype(str).str.match(r"^\d{5}[A-Za-z]*$")
    mask = (~is_cancel & is_product
            & (df["Quantity"] > 0) & (df["Price"] > 0)
            & df["CustomerID"].notna())
    sales = df[mask].copy()
    sales["CustomerID"] = sales["CustomerID"].astype(int)
    return sales

if __name__ == "__main__":
    df = load_raw()
    print("원본:", df.shape)
    sales = load_sales()
    print("정제:", sales.shape)
    print("고객 수:", sales["CustomerID"].nunique())