---
date: 2023-08-14
time: 19:20
note_maturity: 🌱
tags: 
---

# QuickSort

## Divide and Conquer strategy

split your input into chunks and operate over smaller subsets that can be solved easily

## Algorithm

\[                                \]
\0                          p N
^
^

Start with two pointers at the start, and a pivot pointer at the end (or somewhere in the middle)

index 1, incremented every time a value is copied to this index
index 2,  walks the entire array, compares each value against the pivot
- if any value (including the pivot) <= p, then swap that value with the value currently at index 1

after first iteration, pick more pivots and subsets of values, excluding the previous, then repeat

This approach forms a branching structure

The value of this approach is that you can do all your operations in place.

O(n log n)

## Cons / Downsides

Reverse sorted array -> O(n^2)

Is actually between O(n log n) and O(n^2)

## Strategies

you can always pick the middle index











#### 🧭  Idea Compass
- West  (similar) 
[[Algorithms]]
[[Computer Science]]
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
- East (opposite)

- North (theme/question)

- South (what follows)
