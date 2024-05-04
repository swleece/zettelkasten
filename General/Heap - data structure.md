---
date: 2023-10-11
time: 07:47
note_maturity: 🌱
tags:
---
# Heap

A specialized tree-based data structure that satisfies the *heap property*: 
- in a max heap, for any given node C, if P is a parent node of C, then the *key* (the *value*) of P is greater than of equal to the key of C. 
- in a min heap, the key of P is less than or equal to the key of C

The node at the "top" of the heap (with no parents) is called the *root* node.

- can be used to implement a [[Priority Queue]]
- can be implemented with an [[Arraylist]]

Typically you don't traverse the tree.
Whenever you insert or delete, you need to update the tree.

Min heap: root node must be the smallest (l.t. or equal)
Max heap: root node must be the largest (g.t. or equal)

Heaps maintain a weak ordering 

Heaps are full or complete trees (always add left to right, with no empty nodes or gaps)

Insert: add node to next open position, bubble up if needed to satisfy heap condition
Delete: replace deleted node with last node, bubble down to the right position

Can store node values as an array (due to the complete tree nature)
- 2( i ) + 1 , 2( i ) + 2 : formulas for children indexes
- floor( i / 2 ) : formula for parent index 
- keep track of length of arraylist

For updating node values, need to keep a hash map.

log(N) for both operations covered


## Trie (named after Re"trie"val Tree)

also called try trees, prefix tree, digital tree

Can think of it as an auto-complete, O(1)

In english, 26 characters possible, each node can have 26 children theoretically

Each letter is a node, words built as a path
- isWord can be denoted as a boolean for each node (or a child can be added denoting the parent is a word)

insertion : 
deletion : easiest to do recursively




















#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
[[Data Structures]]
- East (opposite)

- North (theme/question)

- South (what follows)
