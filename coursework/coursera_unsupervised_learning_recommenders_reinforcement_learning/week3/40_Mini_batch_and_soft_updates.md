<40>

## Title
Mini batch and soft updates

## Description
Refinements to the deep reinforcement algorithm

#### Mini batch:
- also applicable to supervised algos

basically, when you have a large dataset, chunk it out and run training on each subset sequentially
May mean training with a smaller [[replay buffer]]


#### Soft Updates:
- attempts to prevent Qnew from moving away from the optimal solution 
- instead of replacing Q entirely with Qnew of latest batch, modify it by some percentage

![[Pasted image 20221119123759.png]]


## Additional Notes


## Linked Cards
{{ direct link to another card }}

## Tags
[[Machine Learning]] 
