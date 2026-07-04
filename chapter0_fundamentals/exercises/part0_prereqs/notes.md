# 0.0 Prerequisites Study Note

- Date: Jun 27 2026

## Core Concepts / Knowledge

> Q. What makes neural networks more powerful than basic statistical methods like linear regression?

While stat methods like linear regression only represent the linear relationship between input and output, neural nets can express non-linear and more complex functions and automatically learn by gradient descent method.

> Q. What are the advantages of ReLU activations over sigmoids?

Since sigmoid with the input of large absolute value results in small gradient, it may cause gradient vanishing problem. On the other hand, ReLU back-propagates the gradient in the same extent as the input on positive side, thus ReLU is more robust to gradient vanishing problem.

*Missing point: ReLU is also more computationally efficient!*

> Q. What is the problem in trying to create a neural network using only linear transformations?

Since regardless of the dimension of neural nets, just stacking linear components is identical to the multiplication of many matrices; resulting in another matrix (i.e., linear function). Thus non-linear activations in the middle of neural nets are essential to make the representation power of neural nets stronger.

> Q. Matrices $A$ and $B$ have shapes $(n, m)$ and $(m, l)$. What is the maximum possible rank of the matrix $AB$?

Theoretically $\min(n,m,l)$, practically $m$ since usually $m \ll n,l$.

> Q. What is the expected value and variance of the sum of two independent normally distributed random variables $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$? Are either of these different if they're correlated? (We don't necessarily expect you to be able to derive this kind of result.)

- Expected value: $E[X_1 + X_2] = EX_1 + EX_2 = \mu_1 + \mu_2$
- Variance: $\text{Var}(X_1+X_2) = \text{Var}(X_1) + \text{Var}(X_2) = \sigma_1^2 +\sigma_2^2$ 
($\because$ $X_1$ and $X_2$ are independent, thus uncorrelated)

If $X_1, X_2$ are corelated, the expected value is the same, but we need the covariance term for variance, i.e., $\text{Var}(X_1+X_2) = \text{Var}(X_1) + \text{Var}(X_2) + 2\text{Cov}(X_1,X_2)$


> Q. What is the derivative of quadratic loss $L(x, y) = \frac{1}{2}(x-y)^2$ wrt the input $x$ (assuming all variables are scalars, not vectors)? How about for cross entropy loss $L(x, y) = -(y\log{x} + (1-y)\log{(1-x)})$, assuming that $x \in (0, 1)$ and $y$ is a binary classification label with value either zero or one? What will be the qualitative behaviour of performing gradient descent on $x$ with these loss functions?

$(x-y)$ for MSE loss, 
$$\begin{cases}
-\frac{1}{x} & y=1\\
\frac{1}{1-x} & y=0
\end{cases}$$
for cross-entropy loss.

MSE loss will pull $x$ down if $x > y$, with the extent proportional to the difference between $x, y$. Cross-entropy loss will make $x$ larger if $y=1$, otherwise make $x$ smaller.

> Q. Suppose $P$ is the probability distribution of which word comes next in natural language, and $Q$ is a language model's estimated probability distribution. What will the cross entropy $H(P, Q)$ be if the model is guessing words uniformly? What will the cross entropy be if the model can predict words with the exact right frequency?

$$\begin{aligned}
H(P,Q) &= -\Sigma p(x) \log q(x) \\
&= -\Sigma p(x) \log \frac{1}{|Q|} \\
&= \log |Q|
\end{aligned}$$

Thus, if the model guesses words uniformly, the corss entropy is just the log-cardinality of the distribution, i.e., the logarithm of the size of vocab.

$$\begin{aligned}
H(P,Q) &= -\Sigma p(x) \log q(x) \\
&= -\Sigma p(x) \log p(x) \\
&= H(P)
\end{aligned}$$

If the model predicts words with the exact right freq, the cross entropy is same as the entropy of P.

## Einops, Einsum & tensors

> Q. Broadcasting can be a very easy place to make mistakes, because it's easy to lose track of the exact shape of your tensors involved. As a warm-up exercise, below are some examples of broadcasting. Can you figure out which are valid, and which will raise errors?

```
x = t.ones((3, 1, 5))
y = t.ones((1, 4, 5))

z = x + y
```

- `z.shape = (3,4,5)`
- Broadcasting 0th dim of `y` and 1st dim of `x`

```
x = t.ones((8, 2, 6))
y = t.ones((8, 2))

z = x + y
```

Invalid. Automatic expansion can only occur in 0th dimension. To make this operation valid, we need to append dummy dimension to `y`, i.e., `y.shape = (8,2,1)`.

*Don't be confused!*

```
x = t.ones((8, 2, 6))
y = t.ones((2, 6))

z = x + y
```

- `z.shape = (8,2,6)`
- Prepending dummy dimension to `y` and broadcasting it

```
x = t.ones((10, 20, 30))
y = t.ones((20, 1))

z = x + y
```

- `z.shape = (10, 20, 30)`
- Prepending dummy dimension to `y` and broadcasting 0th and 2nd dim of `y`

```
x = t.ones((4, 1))
y = t.ones((4,))

z = x + y
```

- `z.shape = (4, 4)`
- Prepending dummy dimension to `y` and broadcasting 1st dim of `y`

*Don't be confused!*