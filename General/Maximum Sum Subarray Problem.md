---
date: 2024-05-12
time: 12:30
note_maturity: 🌱
tags:
  - idea
---
# Maximum Sum Subarray Problem

aka maximum segment sum problem

Although this problem can be solved using several different algorithmic techniques, including brute force, divide and conquer, dynamic programming, and reduction to shortest paths, a simple single-pass algorithm known as Kadane's algorithm solves it efficiently.

Kadane's Algorithm:
- runtime complexity is O(n)
- space complexity is O(1)

```Python
def max_subarray(numbers):
    """Find the largest sum of any contiguous subarray."""
    best_sum = - infinity
    current_sum = 0
    for x in numbers:
        current_sum = max(x, current_sum + x)
        best_sum = max(best_sum, current_sum)
    return best_sum
```






#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]
[[Algorithms]]

- East (opposite)

- North (theme/question)

- South (what follows)
