import networkx as nx
from py2neo import Graph


def draw_neo4j_graph(handle="bolt://localhost:7687", auth=("neo4j", None)):
    # Connect to Neo4j
    graph = Graph(handle, auth=auth)

    # Run a Cypher query
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT 100
    """
    result = graph.run(query)

    # Create a NetworkX graph
    G = nx.DiGraph()  # Use DiGraph for directed edges
    for record in result:
        n = record["n"]
        m = record["m"]
        r = record["r"]
        G.add_node(n["name"], label=list(n.labels or ["-"])[0])
        G.add_node(m["name"], label=list(m.labels or ["-"])[0])
        G.add_edge(n["name"], m["name"], type=type(r).__name__)

    # Draw the graph
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=10000)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)

    # Add node labels
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, {node: f"{node}\n({label})" for node, label in node_labels.items()}, font_size=16)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=16)
