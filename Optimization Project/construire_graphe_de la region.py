import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

# Changer le répertoire de travail pour être celui du script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel('instance_1.xlsx', sheet_name=2)
d=df.values.tolist()
d1 = pd.read_excel('instance_1.xlsx', sheet_name=1)
d = df.values.tolist()

assets=dict()
t= d1.values.tolist()
for i in t:
    assets[i[0] ]= (i[1],i[2])
#print (d)

# Parameters
slr = 3
dimension_1 = len(d)
dimension_2 = len(d[1])

region_level = d

# Create graph
G = nx.Graph()
for i in range(dimension_1):
    for j in range(dimension_2):
        G.add_node((i, j))
        if (i, j) in assets.values():
            G.nodes[(i, j)]['color'] = 'yellow'
            G.nodes[(i, j)]['size'] = 400
        else:
            G.nodes[(i, j)]['color'] = 'blue'
            G.nodes[(i, j)]['size'] = 200
            if region_level[i][j]<3:
                G.nodes[(i, j)]['color'] = 'red'
                G.nodes[(i, j)]['size'] = 190
        if i > 0:
            G.add_edge((i, j), (i-1, j))
        if j > 0:
            G.add_edge((i, j), (i, j-1))

print(G)
# Plot graph
node_colors = [G.nodes[node]['color'] for node in G.nodes()]
node_sizes= [G.nodes[node]['size'] for node in G.nodes()]
pos = {(i, j): (j, -i) for i in range(dimension_1) for j in range(dimension_2)}
nx.draw(G, pos=pos, node_color=node_colors , node_size=node_sizes,with_labels=True)

plt.show()
