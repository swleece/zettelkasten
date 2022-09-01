13

## Title
Precision and Recall

## Description
Precision = 
  true positives / # predicted positive

  OR 

  true positives / (true positives + false positives)



Recall = 
  true positives / # actual positives

  OR

  true positives / (true positives + false negatives)


Trading off precision and recall:

Modifying you confidence threshold for predictions will affect precision 
and recall inversely
Consider the costs of false positives vs false negatives in your application.

The F1 score can be used to automatically select a model / threshold.

F1 score =  2 (PR / (P+R))    (harmonic mean of P and R)


## Additional Notes

                       Actual Class

                             |
                    true     |    false
                  positive   |  positive
                             |
 predicted         __________|__________   
  class                      |
                             |
                    false    |    true
                  negative   |  negative
                             |



Precision
  |
1 | 
  |  .
  |       .
  |         .
  |           x
  |            .
  |             .
  |              .
  |              .
  |____________________
     Recall       1


## Linked Cards
{{ direct link to another card }}

## Tags
[[ Machine Learning ]] 
