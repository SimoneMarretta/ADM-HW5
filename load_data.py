from MyGraph import *
import pandas as pd

def load_data():
    ''' Load data from text files to dataframes
    '''
    dataframes = []
    for name in ['USA-road-d.CAL.gr', 'USA-road-t.CAL.gr']:
        df = pd.read_csv(name, sep=' ', header=None, names=['a','node_1','node_2','distance'])
        df.drop(columns=['a'], inplace = True)
        dataframes.append(df)
        
    df = df.copy()
    df['distance'] = 1
    dataframes.append(df)
    
    adj_graphs = []
    for i in range(len(dataframes)):
        adj_graphs.append(dataframe_to_adj_list(dataframes[i]))
     
    return dict(zip(['d', 't', 'n'], dataframes)), dict(zip(['d', 't', 'n'], adj_graphs))

def load_coordinates():
    coordinates = pd.read_csv('USA-road-d.CAL.co', sep=' ', header=None, names=['v','node','Latitude','Longitude'])
    return dict(zip(coordinates['node'], list(zip(coordinates['Latitude'].values, coordinates['Longitude'].values))))

dataframes, adjacency_lists = load_data()
def choose_data(param):
    return dataframes[param], adjacency_lists[param]

coordinates = load_coordinates()