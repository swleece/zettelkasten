<36>

## Title
Bellman Equations

## Description

Requires an ability to calculate the return of behaving optimally from any given state

Is the sum the return from the current state s, plus the return from behaving optimally starting from state s' (state achieved by taking first action a).


## Additional Notes
s : current state
R(s) = requard of current state
a : current action
s' : state you get into  after taking action a
a' : action that you take in state s'
gamma = discount factor

$$ Q(s,a) = R(s) + \gamma * max\_a' Q(s',a') $$

Q(s,a) = R(s) + gamma * max a' of Q(s',a')

![[bellman_equation.png]]
![[Pasted image 20221119105234.png]]
## Linked Cards
{{ direct link to another card }}

## Tags
[[Machine Learning]] 
