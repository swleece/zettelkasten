---
date: 2023-10-14
time: 13:47
note_maturity: 🌱
tags:
---
# Graphs - data structure

## Terminology

cycle: when you start at node x, follow links, and end back at node x (requires three or more nodes)
acyclic: contains no cycles
connected: every node has a path to another node
directed: there is a direction to the connections
undirected: !directed
weighted: edges have a weight associated with them
dag: directed, acyclic graph

- Node also referred to as vertex
- BigO is commonly stated in terms of V (vertices) and E (edges)

- all trees are graphs, not all graphs are trees

Adjacency List vs Adjacency Matrix (often O(V)^2)

adjacency list: list of lists, outer list index maps to node, inner list maps to connecting edges
```
[
  [{to: 1, weight: 10}],
  [],
  [{to:0, weight:5}, {to: 1, weight: 1}]
]
```

adjacency matrix: list of lists, outer list index maps to node, inner list index corresponds to weight of each edge and includes value for every possible connection (which means you must build a VxV matrix)
```
[
  [0, 10, 0],
  [0, 0, 5],
  [1, 0, 1]
]
```

[[Breadth-First Search]] and [[Depth-First Search]] work on graphs as well as trees


## Breadth First Search on an Adjacency Matrix

![[Pasted image 20231017074702.png]]

Breadth First Search Pseudocode:

```
seen [t, f, f, ... ]
prev [-1, ... ]
Q = [0]

do {
  curr = Q.deque();
  if curr = needle
    break
  for c in curr
    if seen continue
    seen[c] = true
    prev[c] = curr
    Q.enqueu(c); 
} while(Q.len);
  prev[needle] = -1 if never found, otherwise will get where it was.


```


## Depth First Search on an Adjacency List

 




























#### 🧭  Idea Compass
- West  (similar) 
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
[[Computer Science]]
[[Data Structures]]


- East (opposite)

- North (theme/question)

- South (what follows)
