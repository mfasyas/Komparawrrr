from graph import Graph

g = Graph(directed=False)
g.generate_random_connected_graph(num_vertices=5, extra_edges=10)

print("Adjacency List:")
g.display()

print("\nConverted Adjacency Matrix:")
matrix, index_map = g.to_adjacency_matrix()
for row in matrix:
    print(row)

print("\nNode index mapping:")
print(index_map)
