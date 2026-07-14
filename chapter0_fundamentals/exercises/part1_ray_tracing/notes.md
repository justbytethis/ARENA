# 0.1 Ray Tracing Study Note

## Ray-Object Intersection

> Is it possible for the solve method to fail? Give a sample input where this would happen.

If the line segment and ray are parallel, there exists no solution. Thus the solve method would fail. For example, let

$$
O = (0,0), D = (3,3), L_1 = (2,1), L_2 = (1,0).
$$

Then given equation is formed as:

$$
\begin{pmatrix}
    3 & 1 \\
    3 & 1 \\
\end{pmatrix}
\begin{pmatrix}
    u \\ v
\end{pmatrix}
=
\begin{pmatrix}
    2 \\ 1
\end{pmatrix}.
$$

No $u$ and $v$ satisfy the equation above.