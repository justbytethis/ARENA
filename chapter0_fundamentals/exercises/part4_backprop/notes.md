# 0.4 Backpropagation Study Note

## Introduction to Backprop

> Question - can you think of a reason it might be important for a node to store a list of all of its parent nodes?

Backpropagation starts from the end node (root node) and proceeds to the leaf node. Thus, after reaching some nodes, parent nodes are exactly ones to process at the next step. Without this information, we are no longer able to proceed anymore.

> Question - can you think of an example function where it would be computationally cheaper to use 'out' than to use 'x'?

It means the case that $\frac{d\text{out}}{dx}$ is easily computed by $\text{out}$. One clear case is $\text{out} = \exp(x)$, since $\text{out} = \exp(x) = \frac{d\text{out}}{dx}$.