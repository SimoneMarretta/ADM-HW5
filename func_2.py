from MyGraph import *

try:
    dataframes
except NameError:
    import load_data

def smartest_network(nodes, data_type):
    ''' Implementation of an algorithm that returns the set of roads (edges) that enable the user to visit all the places. 
    The result is achived via minimum spanning tree.'''
    df, adj = choose_data(data_type)
    graph = MyGraph({'df': df, 'adj': adj })
    nodes = list(nodes)
    print(nodes)
    tree = graph.spanning_tree(nodes[0], set(nodes[1:]))
    nodes = tree.keys()
    return graph, nodes, graph.adj_to_edges(tree)

def visualize_2(graph, input_nodes, nodes, edges, coordinates, depth = 20):
    ''' Visualization for smartest network. Connections between nodes are shown green. 
    Moreover, neighbors and roads between them are shown with some depth specified.
    '''
    nbrs = set()
    for node in nodes:
        graph.neighbours(node, depth, nbrs)
    
    graph.draw({'g': edges}, coordinates, options={'labels_to_show': input_nodes})
    graph.draw({'k': graph.edges_by_nodes(nbrs)}, coordinates, options={'node_size': 1})
    
def func_2():
    input_nodes = set(map(int, input('Write nodes you want to visit').split()))
    data_type = input('Which distance to use (t,d,n)?')
    graph, nodes, edges = smartest_network(input_nodes, data_type)
    visualize_2(graph, input_nodes, nodes, edges, coordinates, 100)