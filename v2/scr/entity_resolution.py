import pandas as pd
from itertools import combinations
from feature_engineering import compute_features

def generate_pairs(df):

    pairs = []

    for i,j in combinations(df.index,2):

        r1 = df.loc[i]
        r2 = df.loc[j]

        features = compute_features(r1,r2)

        score = (
            features["imo_match"]*3 +
            features["mmsi_match"]*2 +
            features["flag_match"] +
            features["name_similarity"]/100
        )

        if score > 3:
            pairs.append((i,j))

    return pairs