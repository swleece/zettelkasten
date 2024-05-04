---
date: 2023-09-30
time: 08:44
note_maturity: 🌱
tags:
---
# Depth-First Search

[[Trees - data structure]]





## Find

- very simple on a binary search tree and similar to quick sort
- recursively move down tree until you find value
- O(h) (height) which will be between in range between log(n) and n

## Insert

- do the same type of traversal as Find until you find a null value and insert the value there

## Delete

- case 1: no child -> can simply delete
- case 2: 1 child -> set parent value to child (like a singly linked list operation)
- case 3: 2 children ->  need to find largest (rightmost) node on the smaller than subtree OR the smallest (leftmost) node on the larger than subtree and replace the target node with that 
	- probably want to track the height of the tree so that you can choose the side that will shrink up the tree
	- which side you choose to delete from will change the shape of the tree



















#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
[[Binary Search]]
- East (opposite)

- North (theme/question)

- South (what follows)
