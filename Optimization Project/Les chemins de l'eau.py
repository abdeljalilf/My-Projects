import networkx as nx
import matplotlib.pyplot as plt
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
slr = 3
dim1 = len(d)
dim2 = len(d[1])

region_level = d
# Set parameters

# Build graph

G = nx.Graph()
for i in range(dim1):
    for j in range(dim2):
        if region_level[i][j] < slr:
            neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1),
                         (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
            for neighbor in neighbors:
                if (0 <= neighbor[0] < dim1 and 0 <= neighbor[1] < dim2
                        and region_level[neighbor[0]][neighbor[1]] < slr):
                    G.add_edge((i, j), neighbor)


for asset in assets:
    G.add_node(asset)
print(G)
poss = {node: node for node in G.nodes()}
l=list(poss)
s={} # source de l'eau
ch = l.copy() #les point qui peuvent transmettre l'eau aux assets
for (i,j) in l:
    if (i,j) not in poss:
        region_level[i][j]=10
        ch.remove((i,j))
    else:
        if i==0 or i== dim1-1 or j==0 or j== dim2-1:
            s[(i,j)]=(i,j)


# Draw graph
pos = {(i, j): (j, dim1-i-1) for i in range(dim1) for j in range(dim2)}
nx.draw_networkx_nodes(G, pos, nodelist=list(s), node_color='r', node_size=60)
nx.draw_networkx_nodes(G, pos, nodelist=assets, node_color='y', node_size=100)
nx.draw(G, pos=pos, with_labels=False, node_size=40)

nx.draw_networkx_edges(G, pos , edge_color='r')
nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif')
plt.axis('off')
plt.show()

