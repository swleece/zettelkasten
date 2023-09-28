---
date: 2023-08-12
time: 14:50
note_maturity: 🌱
tags: 
---

# Queues

A queue is an implementation of a [[Linked List - data structure]]

(A) -> (B) -> (C) -> (D)
head                          tail

FIFO

Nodes are added after the tail, pulled from the head.

Pop operation pulls from the heaD
- move head pointer to (B) : `head = head.next`
- remove link from (A) -> (B)
- return (A)

Uses a singly linked list.

No list traversing involved but limited to head / tail actions.

Constant time operations.

Peek operation: `return this.head?.value`

Often keep track of length with running counter.







#### 🧭  Idea Compass
- West  (similar) 
[[Data Structures]]
[[Linked List - data structure]]
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
- East (opposite)

- North (theme/question)

- South (what follows)
