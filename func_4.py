from MyGraph import *

try:
    dataframes
except NameError:
    import load_data

def shortes_route(df_distance, adj_graph, starting_node, nodes_to_visit, pos):
    d_graph = MyGraph({'df' : df_distance, 'adj': adj_graph})

    #  Traverse graph in order to find disconnected regions
    parts = d_graph.traverse()

    #  Check if nodes connected with each other
    if not d_graph.are_nodes_connected(parts, {starting_node} | nodes_to_visit):
        print("Not possible")
        return
    
    path = d_graph.detour(starting_node, nodes_to_visit)
    edges = d_graph.route_edges(path)
    
    #  Visualization part
    nbrs = set()
    for node in path:
        d_graph.neighbours(node,5, nbrs)

    d_graph.draw(edges, pos, options={'labels_to_show': nodes_to_visit})
    d_graph.draw({'k': d_graph.edges_by_nodes(nbrs)}, pos, options={'node_size': 1})
    
def func_4():
    input_nodes = set(map(int, input('Write nodes you want to visit. First node will be starting node.').split()))
    data_type = input('Which distance to use (t,d,n)?')
    df_distance, adj_graph = choose_data(data_type)
    starting_node = input_nodes[0]
    node_to_visit = input_nodes[1:]
    shortes_route(df_distance, adj_graph, starting_node, node_to_visit, coordinates)