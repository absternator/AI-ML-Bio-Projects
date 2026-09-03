# AI/ML and Bioinformatics Projects

This repository contains various projects related to Artificial Intelligence, Machine Learning, and Bioinformatics. Each project is organized in its own directory with relevant code, data, and documentation.

Therefore, by the chain rule:

$$
\frac{d}{dt} v_\theta(x_t, r, t)
=
\frac{\partial v_\theta}{\partial x}\,u
+
\frac{\partial v_\theta}{\partial r}\cdot 0
+
\frac{\partial v_\theta}{\partial t}\cdot 1
$$

This full pathwise derivative is what `dvdt` represents.

---

## 3. Meaning of the target

You define:

$$
u_{\text{target}}
=
u - (t-r)\,\frac{d}{dt}v_\theta(x_t,r,t)
$$

### Interpretation

- $u$ is the true instantaneous velocity of the linear bridge.
- $(t-r)\,\frac{d}{dt}v_\theta(x_t,r,t)$ corrects for how your model's predicted mean velocity changes as you move along the path.
- So $u_{\text{target}}$ is a corrected supervision target for mean velocity on the interval $[r,t]$.

Then the loss is the MSE between the model prediction and this corrected target.

---

## 4. Why stop-gradient on the target side

You use:

$$
\left\|
v_\theta
-
\operatorname{sg}(u_{\text{target}})
\right\|^2
$$

This is important because $u_{\text{target}}$ itself depends on model outputs through `dvdt`.

Without stop-gradient, gradients would flow through both sides and create second-order-like coupling and instability.

With stop-gradient, optimization becomes:

> "Predict this target."

while treating the target as fixed for that optimization step, which is usually more stable.

---

## 5. Concrete Jacobian/JVP mini example

Suppose a toy model:

$$
f(x,t)
=
\begin{bmatrix}
x_1^2 + t \\
x_1 x_2
\end{bmatrix},
\qquad
x=(x_1,x_2)
$$

The Jacobian with respect to $(x_1,x_2,t)$ is:

$$
J_f
=
\begin{bmatrix}
2x_1 & 0 & 1 \\
x_2 & x_1 & 0
\end{bmatrix}
$$

Pick the direction

$$
(\dot{x}_1,\dot{x}_2,\dot{t})
=
(u_1,u_2,1).
$$

Then the Jacobian-vector product (JVP) is:

$$
J_f
\begin{bmatrix}
u_1 \\
u_2 \\
1
\end{bmatrix}
=
\begin{bmatrix}
2x_1u_1 + 1 \\
x_2u_1 + x_1u_2
\end{bmatrix}.
$$

This is exactly the directional derivative

$$
\frac{d}{dt} f(x_t,t)
$$

along the path whose velocity is

$$
\frac{dx_t}{dt}=u.
$$


$$\tfrac12\left(\log\sigma^2 + \frac{(y-\mu)^2}{\sigma^2}\right)$$

