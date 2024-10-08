import pulp
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os

# Changer le répertoire de travail pour être celui du script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_excel(r'instance_1.xlsx', sheet_name=2)
d=df.values.tolist()
d1 = pd.read_excel(r'instance_1.xlsx', sheet_name=1)
#d = df.values.tolist()
'''slr = 3
dim1 = 9
dim2 = 9
assets = [(0, 8), (2, 6), (4, 6)]
region_level = [
    [8, 8, 7, 7, 6, 6, 7, 6, 2],
    [2, 4, 7, 6, 8, 5, 1, 5, 2],
    [4, 8, 7, 3, 4, 5, 2, 5, 6],
    [6, 8, 2, 6, 5, 2, 8, 8, 7],
    [1, 8, 8, 3, 4, 9, 2, 3, 6],
    [9, 5, 7, 7, 3, 1, 1, 3, 2],
    [5, 3, 2, 5, 3, 4, 2, 1, 1],
    [7, 3, 9, 6, 9, 6, 2, 4, 2],
    [7, 4, 3, 1, 1, 1, 5, 1, 8]
]'''
assets=[]
t= d1.values.tolist()
for i in t:
    assets.append((i[1],i[2]))
"""
assets.append((7,15))
assets.append((10,1))
assets.append((8,3))
assets.append((1,10))
#assets.append((1,8))
#print (assets)"""

# Parameters
slr = 3
dim1 = len(d)
dim2 = len(d[1])

region_level = d

def get_neighbors(point):
    i, j = point
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == dj == 0:
                continue
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= dim1 or nj < 0 or nj >= dim2:
                continue
            if region_level[ni][nj] < slr :
                neighbors.append((ni, nj))
    return neighbors
"""
def get_neighbors(a):
    i,j=a
    N=[]
    neighbors = [(i-1, j-1), (i, j-1), (i+1, j-1), (i+1, j),  (i+1, j+1), (i, j+1),(i-1, j+1),(i-1, j)]
    for neighbor in neighbors:
        if (0 <= neighbor[0] < dim1) and (0 <= neighbor[1] < dim2):
            if region_level[neighbor[0]][neighbor[1]] < slr:
                if i in [0,dim1-1] or j in [0,dim2-1] :
                    if (i,j) in assets: N.append(neighbor)

                    else:
                        if neighbor[0] not in [0,dim1-1] and neighbor[1] not in [0,dim2-1]:
                            N.append(neighbor)
                        else:
                            if neighbor in assets:
                                N.append(neighbor)
                else:
                    N.append(neighbor)

    return N
"""
def find_water_paths(slr, dim1, assets, region_level):
    # Create a list of sources
    sources = []
    for i in range(dim1):
        if region_level[0][i] < slr:
            sources.append((0, i))
        if region_level[dim1-1][i] < slr:
            sources.append((dim1-1, i))
        if region_level[i][dim1-1] < slr:
            sources.append((i, dim1-1))
        if region_level[i][0] < slr:
            sources.append((i, 0))

    # Define a helper function to perform DFS
    def dfs(curr_pos, visited, path):
        # Add current position to path
        path.append(curr_pos)
        #print(get_neighbors(curr_pos))
        # If the current position is an asset, add the path to the list of water paths
        if curr_pos in assets:
            water_paths.append(path)

        # Visit adjacent positions
        for next_pos in get_neighbors(curr_pos):
            if next_pos[0] < 0 or next_pos[0] >= dim1 or next_pos[1] < 0 or next_pos[1] >= dim1:
                continue # Skip out of bounds positions
            if next_pos in visited:
                continue # Skip visited positions
            if region_level[next_pos[0]][next_pos[1]] >= slr:
                continue # Skip positions above sea level
            if curr_pos in assets :
                continue # Skip paths that pass through other assets
            dfs(next_pos, visited.union({next_pos}), path[:])

    # Initialize list of water paths
    water_paths = []

    # Find water paths for each source
    for source in sources:
        dfs(source, {source}, [])

    # Group water paths by asset and source
    asset_source_paths = {}
    for path in water_paths:
        for asset in assets:
            if asset == path[-1]:
                key = (asset, path[0])
                break
            elif asset == path[0]:
                key = (asset, path[-1])
                break
        if key in asset_source_paths:
            asset_source_paths[key].extend(path)
        else:
            asset_source_paths[key] = path

    # Remove duplicates from grouped paths
    for key in asset_source_paths:
        asset_source_paths[key] = list(dict.fromkeys(asset_source_paths[key]))

    return asset_source_paths


water_paths = find_water_paths(slr, dim1, assets, region_level)

def list_utile():
    l=[]
    for i in find_water_paths(slr, dim1, assets, region_level):
        for j in water_paths[i]:
            if j not in l: l.append(j)
    return l

def getneighbors(a):
    m = list_utile()
    i,j=a
    N=[]
    neighbors = [(i-1, j-1), (i, j-1), (i+1, j-1), (i+1, j),  (i+1, j+1), (i, j+1),(i-1, j+1),(i-1, j)]
    for neighbor in neighbors:
        if (0 <= neighbor[0] < dim1) and (0 <= neighbor[1] < dim2):
            if region_level[neighbor[0]][neighbor[1]] < slr and neighbor in m :
                if i in [0,dim1-1] or j in [0,dim2-1] :
                    if (i,j) not in assets:
                        if neighbor[0] not in [0,dim1-1] and neighbor[1] not in [0,dim2-1]:
                            N.append(neighbor)
                        else:
                            if neighbor in assets:
                                N.append(neighbor)
                    else:N.append(neighbor)
                else:
                    N.append(neighbor)

    return N



m=[(19, 3), (18, 2), (17, 1), (18, 3), (12, 0), (12, 1), (12, 2), (13, 1), (13, 2), (13, 3), (13, 0)]
k=[]

def f(a, neighbors,queue):
    i, j = a
    #print(neighbors)
    min_cost = 0
    bar=[]

    if (i in [0,dim1-1] or j in [0,dim2-1]) and (i,j) not in assets:
        di=[f"bar_{i}_{j}"]
        return (slr-region_level[i][j]),di
    else:
        if (i in [0,dim1-1] or j in [0,dim2-1]) and (i,j) in assets:
            min_cost = (slr-region_level[i][j])
            bar=[f"bar_{i}_{j}"]
        for b in neighbors:
            di=[]
            k,l=b
            queue.append(b)

            new_neighbors = getneighbors(b)
            for d in queue:
                if d in new_neighbors:
                    new_neighbors.remove(d)
            cost = min(slr - min(region_level[i][j], region_level[b[0]][b[1]]), f(b, new_neighbors,queue)[0])
            min_cost = min_cost + cost
            if slr - min(region_level[i][j], region_level[b[0]][b[1]])== cost:
                di=[f"bar_{i}_{j}_{k}_{l}"]
            bar=bar+di
            #print(queue)
            v = queue.pop(0)

        return min_cost,bar
"""
#neighbors=get_neighbors((1,8))
#print(neighbors)
h = f((17,1), getneighbors((17,1)) ,[(17,1)])
print(h)
#print(len(k))
"""
cout=0
bars=[]
for asset in assets:
    cout=cout+f(asset, getneighbors(asset),[asset])[0]
    bars=bars+f(asset, getneighbors(asset),[asset])[1]
print(cout)

print(bars)

