import networkx as nx

def build_graph(pairs):

    G = nx.Graph()

    for p in pairs:
        G.add_edge(p[0],p[1])

    return G


def get_clusters(G):

    clusters = list(nx.connected_components(G))

    return clusters