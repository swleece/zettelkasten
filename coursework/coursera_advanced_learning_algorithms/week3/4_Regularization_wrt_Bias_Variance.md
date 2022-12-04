4

## Title
Regularization wrt bias and variance

## Description
very large lambda -> w parameter very small -> high bias (underfit)
very small lambda -> effectively no regularization -> high variance (overfit)

Choosing a regularization parameter:
create training set, cross validation set, test set
Train multiple models with range of regularization parament (e.g. 0 - 10)
Pick best one based on cost associated with cv set.
Get generalizability by computing cost associated with the test set.

## Additional Notes

x - Jcv
. - Jtrain

 | x                x .
 |  x              x  .
 |   x            x  .
 |     x       x    .
J|         x       .
 |                .
 |              .
 |           .
 | .    .       
 |                     
 _______________________
  lambda



## Linked Cards
{{ direct link to another card }}

## Tags
[[Machine Learning]] 
