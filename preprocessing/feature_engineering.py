import pandas as pd
import numpy as np

def check_data_health(df: pd.DataFrame, name: str = "Data"):
    print(f"\n=== {name} Summary ===")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}\n")
    print(f"Data is: {df.head(10)}\n")
    return df

def handle_missing_and_duplicates(df: pd.DataFrame, name: str = "Data") -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.ffill().bfill() 
    print(f"\n=== {name} Cleaned ===")
    print(f"Remaining missing values: {df.isnull().sum().sum()}\n")
    print(f"Data is: {df.head(10)}\n")
    return df

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    return df

def add_lag_features(df: pd.DataFrame, lags: list[int], group_col: str = "id") -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: list[int], group_col: str = "id") -> pd.DataFrame:
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(window).mean()
        )
    return df

def merge_external_data(df: pd.DataFrame, oil: pd.DataFrame,  
                        transactions: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    if 'dcoilwtico' not in df.columns:
        df = df.merge(oil, on="date", how="left")
        print("Shape after oil merge:", df.shape)
    if 'transactions' not in df.columns:
        df = df.merge(transactions, on=["date", "store_nbr"], how="left")
        print("Shape after transactions merge:", df.shape)
    if 'city' not in df.columns:
        df = df.merge(stores, on="store_nbr", how="left")  
        print("Shape after city merge:", df.shape)
    df = check_data_health(df, "Merged Data")
    df = handle_missing_and_duplicates(df, "Merged Data")
    if "sales" in df.columns:
        df["log_sales"] = np.log1p(df["sales"])
        # df = df.dropna(subset=['lag_7', 'lag_14', 'rolling_mean_7'])

    df = df.fillna(0)  
    print("Missing value-Final:", df.isnull().any())
    print("Shape of the data is:", df.shape)

    return df

def add_holiday_feature(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    if 'is_holiday' not in df.columns:
        holidays_df = holidays_df[holidays_df['transferred'] == False]
        holidays_df = holidays_df[['date']].copy()
        holidays_df['is_holiday'] = 1

        df = df.merge(holidays_df, on='date', how='left')
        print("The merged columns with holiday is:", df.columns)
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    return df

def prepare_features(df: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame,
                     transactions: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    
    oil = check_data_health(oil, "Oil Data")
    oil = handle_missing_and_duplicates(oil, "Oil Data")
    

    holidays = check_data_health(holidays, "Holiday Data")
    holidays = handle_missing_and_duplicates(holidays, "Holiday Data")
 

    transactions = check_data_health(transactions, "Transactions Data")
    transactions = handle_missing_and_duplicates(transactions, "Transactions Data")

    stores = check_data_health(stores, "Stores Data")
    stores = handle_missing_and_duplicates(stores, "Stores Data")

    df = check_data_health(df, "Training Data")
    df = handle_missing_and_duplicates(df, "Training Data")

   
    df = create_date_features(df)
    df = add_holiday_feature(df, holidays)
    df = merge_external_data(df, oil, transactions, stores)
    df = check_data_health(df, "After merging Data")
    df = add_lag_features(df, lags=[7, 14], group_col="id")
    df = add_rolling_features(df, windows=[7], group_col="id")

    df = check_data_health(df, "Final Feature Data")
    # df = handle_missing_and_duplicates(df, "Final Feature Data")

    if "sales" in df.columns:
        df["log_sales"] = np.log1p(df["sales"])
        df = df.dropna(subset=['lag_7', 'lag_14', 'rolling_mean_7'])
    else:
    # In inference, fill lag/rolling with 0 since no sales history
        for col in ['lag_7', 'lag_14', 'rolling_mean_7']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    df = df.fillna(0)
    print("Missing value-Final:", df.isnull().any())
    print("Shape of the data is:", df.shape)
    return df
