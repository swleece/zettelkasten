---
date: 2023-08-13
time: 12:21
note_maturity: 🌱
tags: 
---

# ArrayBuffer (or Ring Buffer)

\[                                    \]
         ^            ^
\0        h             t         n

items ahead of head and after tail are null

Enables pushing, popping, enqueuing, dequeing in O(1)

`this.tail % len` used to take pointer back around to the front
- so it can work in a loop effectively, there's only an issue if you use up all possible positions
- requires a resize (when tail and head would be overlapping)

## Example

Could be useful for something like logs where you need to constantly be adding and also flush values sometimes 













#### 🧭  Idea Compass
- West  (similar) 
[[Data Structures]]
[[The Last Algorithms Course You'll Need - ThePrimeagen]]
- East (opposite)

- North (theme/question)

- South (what follows)
