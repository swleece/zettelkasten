<31>

## Title
Implementing collaborative filtering using tensorflow

## Description
![[collaborative_filtering_tf_implementation.png]]

## Additional Notes
```tf.Variable(3.0)``` : tf.variables are parameters we want to optImize
```tf.GradientTape() as tape:```  <- gradient tape syntax

![[tensorflow_gradient_descent_calculus.png]]

Drawbacks:
- Suffers from cold start problem, hard to rank new items that few users have rated.
	- when you have a new user, results may not be accurate
- Doesn't use side information about items or users that you may have access to
	- e.g. genre, actors, studio or user demographics, location etc...



## Linked Cards
{{ direct link to another card }}

## Tags
[[Machine Learning]] 
