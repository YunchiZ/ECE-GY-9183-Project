def clean_text(df):
    df = df.drop_duplicates(subset=["headline", "short_description"])
    df = df.dropna(subset=["headline", "short_description", "category"])
    return df

def clean_text_welfake(df):
    df.dropna(inplace=True)
    return df

def clean_text_summary(df):
    df.dropna(inplace=True)
    return df
