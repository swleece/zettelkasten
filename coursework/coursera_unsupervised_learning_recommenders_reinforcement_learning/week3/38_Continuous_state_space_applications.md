<38>

## Title
Continuous State space applications

## Description
continuous state in contrast with discrete state

e.g.
$$ s = [x pos, y pos, orientation, velocity x, velocity y, velocity \theta] $$
and roll, pitch, yaw, and their rates of change for helicopter


## Additional Notes


## Examples
previous mars rover, actual truck, helicopter, lunar lander 

#### Lunar Lander: Deep Reinforcement Learning

Q(s,a) = y (target value, which we train the NN to approximate)

Input is state-action pair, Output is value of that state-action pair, maximize that

![[Pasted image 20221119120124.png]]
![[Pasted image 20221119121402.png]]
Replay buffer : only store the 10,000 most recent tuples (returns) while training 

#### Refinement

most implementations use slightly different NN architecture

instead of calculating inference of all four possible actions (Q(s,nothing), Q(s,left), Q(s,right), Q(s, main))

more efficient to train a single NN with 4 output units in output layer corresponding to the 4 possible actions 

![[Pasted image 20221119121844.png]]


## Linked Cards
[[33_Reinforcement_learning_overview]]

## Tags
[[Machine Learning]] 
#DQNAlgorithm

