from rapidfuzz import fuzz

def compute_features(row1,row2):

    features = {}

    features["imo_match"] = int(row1["imo"] == row2["imo"])
    features["mmsi_match"] = int(row1["mmsi"] == row2["mmsi"])
    features["flag_match"] = int(row1["flag"] == row2["flag"])

    features["name_similarity"] = fuzz.ratio(
        row1["name"],
        row2["name"]
    )

    return features