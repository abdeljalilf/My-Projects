import pulp
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os

# Changer le répertoire de travail pour être celui du script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_excel(r'instance_2.xlsx', sheet_name=2)
d=df.values.tolist()
d1 = pd.read_excel(r'instance_2.xlsx', sheet_name=1)
#d = df.values.tolist()

assets=[]
t= d1.values.tolist()
for i in t:
    assets.append((i[1],i[2]))

# Parameters
slr = 3
dim1 = len(d)
dim2 = len(d[1])

region_level = d
# Set parameters
links = {}
# Define a function to get the neighbors of a point

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
                var_name = f"Link_{i}_{j}_{ni}_{nj}"
                links[(i, j, ni, nj)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)
    return neighbors

sources = []
for i in range(dim2):
    if region_level[0][i] < slr  :
        sources.append((0, i))
    if region_level[dim1-1][i] < slr :
        sources.append((dim1-1, i))
for j in range(dim1):
    if region_level[j][dim2-1] < slr :
        sources.append((j, dim2-1))
    if region_level[j][0] < slr :
        sources.append((j, 0))

#print(water_paths)
# Build graph and  Define binary variables for each link of the graph

G = nx.Graph()
B = nx.Graph()
for i in range(dim1):
    for j in range(dim2):
        B.add_node((i,j))
        if region_level[i][j] < slr:
            for neighbor in get_neighbors((i,j)):

                G.add_node((i,j))
                #G.add_edge((i, j), neighbor)

for asset in assets:
    G.add_node(asset)

# Define a function to find all paths from a source to all assets
def find_paths_from_source(source):
    paths = {}
    queue = deque([(source, [source])])
    while queue:
        point, path = queue.popleft()
        for neighbor in get_neighbors(point):
            if neighbor not in path:
                new_path = path + [neighbor]
                if neighbor in assets:
                    if neighbor not in paths:
                        paths[neighbor] = []
                    paths[neighbor].append(new_path)
                else:
                    queue.append((neighbor, new_path))
    return paths


# Find all paths from each source to each asset
paths = {}

for source in sources:
    paths[source] = {}
    for asset in assets:
        asset_paths = find_paths_from_source(source)
        if asset in asset_paths:
            paths[source][asset] = asset_paths[asset]

# Create optimization model
prob = pulp.LpProblem("Protecting over sea level rise", pulp.LpMinimize)
# Define objective function
prob += pulp.lpSum([links[(i, j, k, l)] *(slr - min(region_level[i][j] , region_level[k][l])) for i, j, k, l in links])
print(len(links))
# Add constraints
#   1 paths[s][a]

for s in sources:
    for a in assets:
        if a in paths[s]:
            for ch in paths[s][a]:
                prob += pulp.lpSum([links[(ch[i][0],ch[i][1], ch[i+1][0],ch[i+1][1])] for i in range (len(ch)-1) ]) == 1

# Solve optimization problem
prob.solve()
#print(links)

###
# Add edges to the graph based on the binary variables
for (i, j, k, l), var in links.items():
    if var.value() == 1:
        G.add_edge((i, j), (k, l))
# Draw graph of connections
pos = {(i, j): (j, -i) for i in range(dim1) for j in range(dim2)}
nx.draw(B, pos=pos, node_color='lightblue', node_size=15 , )
nx.draw(G, pos=pos, node_color='red', node_size=30, )#with_labels=True
nx.draw_networkx_nodes(G, pos=pos, nodelist=assets, node_color='y', node_size=50)
nx.draw_networkx_edges(G,pos=pos , edge_color='g', width=5, )
nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif')
plt.title("Graph of connections")
#plt.show()

# Draw graph of barriers
barrier_pos = [pos[i] for i in assets]
nx.draw_networkx_edges(G, pos=pos, edgelist=[], edge_color='red', width=40)
plt.title("Graph of barriers")
plt.show()