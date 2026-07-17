import yfinance as yf
import pandas as pd
from datetime import datetime
import os

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
START_DATE = "2019-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RAW_DATA_DIR = "."

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Fetching data for {ticker} ...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol or connection.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df["Ticker"] = ticker
    return df

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    all_data = []
    for ticker in TICKERS:
        try:
            df = fetch_stock_data(ticker, START_DATE, END_DATE)
            out_path = os.path.join(RAW_DATA_DIR, f"stock_raw_{ticker}.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved {len(df)} rows -> {out_path}")
            all_data.append(df)
        except Exception as e:
            print(f"FAILED for {ticker}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(RAW_DATA_DIR, "stock_raw_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined raw dataset saved -> {combined_path} ({len(combined)} rows)")

if __name__ == "__main__":
    main()