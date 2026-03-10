import pandas as pd

def validate_imo(imo):
    imo = str(imo)
    if len(imo) != 7:
        return False
    
    digits = [int(d) for d in imo]
    checksum = sum((7-i)*digits[i] for i in range(6)) % 10
    
    return checksum == digits[6]


def clean_data(path):

    df = pd.read_csv(path)

    df = df.dropna(subset=["imo","mmsi","name"])

    df["valid_imo"] = df["imo"].apply(validate_imo)

    df = df[df["valid_imo"] == True]

    return df