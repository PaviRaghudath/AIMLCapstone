import pandas as pd

def load_all_data(data_dir="backend/data"):
    
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["date"])
    holidays = pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(f"{data_dir}/oil.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    transactions = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["date"])

    

    return {
        "train": train,
        "test": test,
        "holidays": holidays,
        "oil": oil,
        "stores": stores,
        "transactions": transactions
    }
