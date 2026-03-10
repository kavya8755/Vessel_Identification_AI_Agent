def search_vessels(df, query):

    query = query.lower()

    results = df[
        df["name"].str.lower().str.contains(query)
    ]

    return results.to_dict(orient="records")