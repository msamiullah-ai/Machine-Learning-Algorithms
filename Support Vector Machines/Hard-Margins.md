# Linear Hard-Margin SVM

In **Linear Hard-Margin SVM**, SVM has to pick a **hard decision boundary**, meaning it does **not allow any training point to be misclassified**. Its goal is to find the separating hyperplane that has the **maximum margin** from both classes. This process only works when the data is **perfectly linearly separable**. If the data is not linearly separable, we usually use a **kernelized SVM with soft margins**.

Suppose a candidate hyperplane is given by

$$
w^T x + b = 0.
$$

First, we ensure that every training point is classified correctly. The prediction is made using the sign of $w^T x + b$:

- If $w^T x + b \ge 0$, the point is classified as the **positive class**.
- Otherwise, it is classified as the **negative class**.

Instead of writing two separate conditions, SVM combines them into a single constraint:

$$
y_i(w^T x_i + b) \ge 0.
$$

This simply means every training point must lie on the correct side of the hyperplane.

---

## Step 1: Define the Margin

Our goal is to **maximize the margin**. The geometric margin is the distance from the hyperplane to the **closest training point**, because the closest point limits how large the margin can be.

Therefore,

$$
\gamma = \min_i \frac{|w^T x_i + b|}{\|w\|}.
$$

Here:

- $\|w\|$ is the **length (magnitude) of the normal vector** $w$.
- The minimum is taken over all training points because we care about the **closest** one.

Since $\|w\|$ is the same for every training point, it can be taken outside the minimum:

$$
\gamma = \frac{1}{\|w\|}\min_i |w^T x_i + b|.
$$

---

## Step 2: Use Scale Invariance

At this point, the expression is still difficult to optimize because of the minimum term.

Now we use an important property called **scale invariance**. The hyperplane does **not** change if we multiply both $w$ and $b$ by the same positive constant.

So, instead of changing the hyperplane, we simply choose the scale of $w$ and $b$ such that the closest training points satisfy

$$
|w^T x_i + b| = 1.
$$

This is equivalent to writing the constraint as

$$
y_i(w^T x_i + b) \ge 1.
$$

After this normalization,

$$
\min_i |w^T x_i + b| = 1,
$$

so the margin becomes

$$
\gamma = \frac{1}{\|w\|}.
$$

---

## Step 3: Convert to an Optimization Problem

Now our objective is maximizing margin, γ whose value we'll substitute as; 

$$
\max \frac{1}{\|w\|},
$$

which is exactly equivalent to

$$
\min \|w\|,
$$

because making $\|w\|$ smaller makes $\frac{1}{\|w\|}$ larger.

Finally, instead of minimizing $\|w\|$, SVM minimizes

$$
w^T w,
$$

because

$$
\|w\| = \sqrt{w^T w}.
$$

Both objectives give the **same optimal solution**, but minimizing $w^T w$ is mathematically much easier since it removes the square root.

---

## Final Hard-Margin SVM Optimization Problem

Therefore, the final Hard-Margin SVM optimization problem becomes

$$
\min_{w,b} w^T w
$$

subject to

$$
y_i(w^T x_i + b) \ge 1.
$$

Thus, Maximizing the margin simply requires us to minimize the length of the weight vector w. By solving this constrained optimization problem, SVM finds the hyperplane with the **maximum possible margin**, and the closest training points (called **support vectors**) are the ones that determine the position of the optimal hyperplane.
