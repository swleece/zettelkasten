11

## Title
Transfer Learning: using data from a different task

## Description
If your data set is small, train a neural network on a similar task against 
a different, very large dataset.
(for example: image classification of cats/dogs -> letters)
Then remove the output layer and replace it with an appropriate one for your
data.
With the parameters initialized by the other dataset, then run training. 
Two options:
1. only train the new output layer (prob only if you have very small dataset)
2. train all parameters 

Need the same input type across two datasets

In practice, usually you are downloading someone else's pretrained parameters
that they have published. Then you can fine tune the network on you own data.

## Additional Notes
This method would be described as supervised pretraining followed by fine 
tuning.

Why does it work? 
Neurons in hidden layers are observed to correspond to identifying edges in 
one layer for example, corners in another, curves in another layer.
This transfers to letter recognition.

## Linked Cards
{{ direct link to another card }}

## Tags
[[ Machine Learning ]] 
