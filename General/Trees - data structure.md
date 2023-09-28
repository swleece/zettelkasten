---
date: 2023-09-16
time: 20:48
note_maturity: 🌱
tags:
---

# Trees - data structure

examples 
- the filesystem is a tree
- the dom is a tree
- very important in compilers (Abstract Syntax Tree)
```
node<T>
	value: T
	children: []
```

## Terminology

- root - most parent
- height - longest path from root to most child
- binary tree - tree in which nodes have at most 2 children, at least 0 children
- general tree - a tree with 0 or more children
- binary search tree - a tree that has a specific ordering to the nodes and at most 2 children
- leaves - a node without children
- balanced - *perfectly* balanced when any node's left and right children have the same height
- branching factor - the amount of children a tree has, per node

## Tree Traversal 

where you attempt to visit every node in a tree
### Binary Tree traversal

pre order traversal - root gets placed at start
1. visit node
2. do something with node value
3. recurseLeft()
4. recurseRight()

in order - root gets placed in middle
1. visit node
2. recurse()

post order - root gets placed at end
1. visit node
2. recurseL()
3. recurseR()
4. visit node

linear complexity - O(n)
all depth first searches 
- stack-like
- also preserve shape
- 

 
### Breadth-First Search

O(n)
- but when using a JS array, it's O(n^2), due to shift and unshift being O(n)
- for this reason you really should use a real queue

queue-like 
visits the tree each level at a time, appending children to the queue to be visited in turn

#### Implementation

Don't need to use recursion
















#### 🧭  Idea Compass
- West  (similar) 
[[Data Structures]]
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
- East (opposite)

- North (theme/question)

- South (what follows)
