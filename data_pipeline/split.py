from sklearn.model_selection import train_test_split

def split_data(df, val_size=0.2, late_ratio=0.1, seed=42):
    df_main, df_late = train_test_split(df, test_size=late_ratio, random_state=seed)
    df_train, df_val = train_test_split(df_main, test_size=val_size, random_state=seed)
    print(f"Train size: {len(df_train)}")
    print(f"Val size: {len(df_val)}")
    print(f"Late data size: {len(df_late)}")
    
    return df_train, df_val, df_late
