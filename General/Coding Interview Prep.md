---
date: 2025-07-27
time: 07:43
note_maturity: 🌱
tags:
  - project
---

# Coding Interview Prep

## Algorithms

### Graph

Graph problems involve:
- nodes (vertices) and edges (directed or undirected) connecting them
- weights on edges
- special structures: trees (acyclic, directed), DAGs (directed, acyclic), grids (implicit 2D graphs)

Typical problem tasks include:
- traversal: "is there a path from A to B?"
- connectivity: "is graph fully connected?"
- shortest path: "minimum cost/steps between two nodes"
- cycle detection
- topological sort: order nodes in a DAG so that earlier nodes come before later
- Minimum Spanning Tree (MST): connect all nodes with minimal total edge weight
- Other specialized: 
	- bipartiteness check (no edges between vertices within the same set)
	- network flow
	- strongly connected components

Two common ways to store a graph in code:

Adjacency List:
```python
# For an undirected graph
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0],
    3: [1]
}
```

Adjacency Matrix:
```python
# V x V matrix, where matrix[u][v] = weight or 1 if there’s an edge
matrix = [[0,1,1,0],
          [1,0,0,1],
          [1,0,0,0],
          [0,1,0,0]]
```