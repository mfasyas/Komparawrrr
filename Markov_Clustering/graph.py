import random

class Graph:
    def __init__(self, directed=False):
        self.graph = {}  # adjacency list
        self.directed = directed

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, src, dest):
        self.add_node(src)
        self.add_node(dest)
        if dest not in self.graph[src]:
            self.graph[src].append(dest)
        if not self.directed and src not in self.graph[dest]:
            self.graph[dest].append(src)

    def generate_fully_connected_graph(self, num_vertices, allow_self_loops=False):
        for i in range(num_vertices):
            self.add_node(i)
        for i in range(num_vertices):
            for j in range(num_vertices):
                if not allow_self_loops and i == j:
                    continue
                if self.directed or i < j:
                    self.add_edge(i, j)

    def generate_random_connected_graph(self, num_vertices, extra_edges=0):
        for i in range(num_vertices):
            self.add_node(i)

        nodes = list(range(num_vertices))
        random.shuffle(nodes)
        for i in range(1, num_vertices):
            src = nodes[i]
            dest = random.choice(nodes[:i])
            self.add_edge(src, dest)

        possible_edges = set()
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i == j:
                    continue
                if not self.directed and i > j:
                    i, j = j, i
                if j not in self.graph[i]:
                    possible_edges.add((i, j))

        extra_edges = min(extra_edges, len(possible_edges))
        for src, dest in random.sample(list(possible_edges), extra_edges):
            self.add_edge(src, dest)

    def to_adjacency_matrix(self):
        nodes = sorted(self.graph.keys())
        index_map = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        matrix = [[0] * size for _ in range(size)]

        for src in nodes:
            for dest in self.graph[src]:
                i = index_map[src]
                j = index_map[dest]
                matrix[i][j] = 1

        return matrix, index_map  # Returning map helps track original node labels

    def display(self):
        for node in self.graph:
            print(f"{node} -> {self.graph[node]}")
