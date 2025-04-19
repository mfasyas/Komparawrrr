import random

class Graph:
    def __init__(self, directed=False):
        self.graph = {}  # Adjacency list
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

    def generate_random_connected_graph(self, num_nodes, num_edges):
        if num_edges < num_nodes - 1:
            raise ValueError("Not enough edges to ensure connectivity.")
        
        max_edges = num_nodes * (num_nodes - 1) if self.directed else num_nodes * (num_nodes - 1) // 2
        if num_edges > max_edges:
            raise ValueError("Too many edges for given number of nodes.")

        # Step 1: Add all nodes
        for i in range(num_nodes):
            self.add_node(i)

        # Step 2: Build a random spanning tree (ensures connectivity)
        nodes = list(self.graph.keys())
        random.shuffle(nodes)
        for i in range(1, num_nodes):
            src = nodes[i]
            dest = random.choice(nodes[:i])
            self.add_edge(src, dest)

        # Step 3: Add remaining edges randomly
        existing_edges = set()
        for src in self.graph:
            for dest in self.graph[src]:
                if not self.directed:
                    edge = tuple(sorted((src, dest)))
                else:
                    edge = (src, dest)
                existing_edges.add(edge)

        potential_edges = set()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if not self.directed:
                    edge = tuple(sorted((i, j)))
                else:
                    edge = (i, j)
                if edge not in existing_edges:
                    potential_edges.add(edge)

        extra_edges = num_edges - (num_nodes - 1)
        new_edges = random.sample(list(potential_edges), extra_edges)
        for src, dest in new_edges:
            self.add_edge(src, dest)
            
    def to_adjacency_matrix(self):
        nodes = sorted(self.graph.keys())
        index_map = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        matrix = [[0] * size for _ in range(size)]

        for src in self.graph:
            for dest in self.graph[src]:
                i = index_map[src]
                j = index_map[dest]
                matrix[i][j] = 1

        return matrix, index_map

    def display(self):
        for node in self.graph:
            print(f"{node} -> {self.graph[node]}")
