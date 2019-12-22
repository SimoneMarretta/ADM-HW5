# ADM-HW5
This repository contain:
* `main.py`: the main python file.
* `load_data.py` : script to load data
* `MyGraph.py`: Graph class with functionalities for task 2 and 4
* files: `func_1.py`, `func_2.py`, `func_3.py`, `func_4.py`

## Main.py once executed has four functionalities:
  ## func_1
  It takes in input:

    -a node v

    -One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).
    a distance threshold d

  It returns the set of nodes at distance <= d from v, corresponding to v’s neighborhood.
  ## func_2
  It takes in input:

    -a set of nodes v = {v_1, ..., v_n}

    -One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).

  It returns the set of roads (edges) that enable the user to visit all the places. We want this set to be the ones whose sum of distances is         minimum.
  ## func_3
  It takes in input:

    -a node H

    -A sequence of nodes p = [p_1, ..., p_n]

    -One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).

  It returns the shortest walk that goes from H to p_n, and that visits in order the nodes in p.
  We used Dijkstra algorithm with a priority queue.
  ## func_4
  It takes in input:

    -a node H

    -A set of nodes p = {p_1, ..., p_n}

    -One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).

  It returns the shortest walk that goes from H to p_n, and that visits the nodes in p.
