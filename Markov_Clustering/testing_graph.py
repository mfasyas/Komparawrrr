from graph import Graph

g = Graph(directed=False)
g.generate_random_connected_graph(num_nodes=5, num_edges=10)

print("Adjacency List:")
g.display()

print("\nAdjacency Matrix:")
matrix, index_map = g.to_adjacency_matrix()
for row in matrix:
    print(row)

print("\nNode-to-Index Map:")
print(index_map)
