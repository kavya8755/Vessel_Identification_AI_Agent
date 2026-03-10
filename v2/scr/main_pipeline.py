from data_cleaning import clean_data
from entity_resolution import generate_pairs
from graph_cluster import build_graph,get_clusters

df = clean_data("data/sample_vessels.csv")

pairs = generate_pairs(df)

G = build_graph(pairs)

clusters = get_clusters(G)

print("Resolved Vessel Clusters")

for c in clusters:
    print(c)