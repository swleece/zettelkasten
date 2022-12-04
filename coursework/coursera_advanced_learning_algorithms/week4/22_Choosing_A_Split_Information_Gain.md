22

## Title
Choosing splits for information gain

## Description
- reduction of entropy is called information gain
- goal is to reduce entropy as much as possible

Use a weighted average the entropy calculation by count of examples.

Information Gain = Entropy at root node - entropy reduced by a splitting on a given feature

p1 left = fraction of positive examples that went to left sub-branch
wleft = fraction of examples that went to left sub-branch
... right

Information Gain =
  H(p1 root) - (wleft * H(p1 left) + wright * H(p1 right)

Selecting a feature that maximizes information gain

## Additional Notes


## Linked Cards
{{ direct link to another card }}

## Tags
[[Machine Learning]] 
