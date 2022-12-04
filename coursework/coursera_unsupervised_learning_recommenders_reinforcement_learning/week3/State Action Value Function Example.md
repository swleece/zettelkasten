---
date: 2022-11-22
tags: JupyterNotebook
---
[[Machine Learning]]

# State Action Value Function Example

In this Jupyter notebook, you can modify the mars rover example to see how the values of Q(s,a) will change depending on the rewards and discount factor changing.


```python
import numpy as np
from utils import *
```


```python
# Do not modify
num_states = 6
num_actions = 2
```


```python
terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = -10

# Discount factor
gamma = 0.9

# Probability of going in the wrong direction
misstep_prob = 0.5
```


```python
generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob)
```


![[Pasted image 20221122211625.png]]