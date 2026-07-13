# Linear Soft-Margin SVM

In **Linear Soft-Margin SVM**, SVM still tries to find a decision boundary with the **maximum possible margin**, but unlike Hard-Margin SVM, it **allows some training points to violate the margin or even be misclassified**. This makes Soft-Margin SVM suitable for **non-linearly separable data**, noisy datasets, and datasets containing outliers. In practice, Soft-Margin SVM is used much more often than Hard-Margin SVM.

Suppose our separating hyperplane is

$$
w^T x + b = 0.
$$

where

- $w$ = normal vector to the hyperplane
- $b$ = bias

---

## Step 1: Recall Hard-Margin SVM

In Hard-Margin SVM, every training point had to satisfy

$$
y_i(w^T x_i + b) \ge 1.
$$

This means every point must

- be classified correctly.
- lie on or outside the margin.

No exceptions are allowed.

This works only when the data is **perfectly linearly separable**.

---

## Step 2: The Problem with Hard Margin

Real-world datasets are rarely perfectly separable.

Sometimes

- classes overlap.
- there is measurement noise.
- there are outliers.

If we still force

$$
y_i(w^T x_i + b) \ge 1,
$$

there may be **no hyperplane that satisfies every point**.

So instead of forcing every point to satisfy the constraint exactly, we allow some flexibility.

---

## Step 3: Introduce Slack Variables

For every training point, SVM introduces a **slack variable**

$$
\xi_i \ge 0.
$$

The constraint now becomes

$$
y_i(w^T x_i + b) \ge 1 - \xi_i.
$$

The slack variable measures **how much a point violates the margin**.

Each training point has its own slack variable.

---

## Step 4: Meaning of Different Values of $\xi_i$

### Case 1: $\xi_i = 0$

Then

$$
y_i(w^T x_i + b) \ge 1.
$$

The point satisfies the margin perfectly.

- Correctly classified ✅
- Outside the margin ✅
- No violation ✅

---

### Case 2: $0 < \xi_i < 1$

For example,

$$
\xi_i = 0.3.
$$

Then

$$
y_i(w^T x_i + b) \ge 0.7.
$$

The point is still on the correct side of the hyperplane but lies inside the margin.

- Correctly classified ✅
- Margin violated ✅

---

### Case 3: $\xi_i = 1$

Then

$$
y_i(w^T x_i + b) \ge 0.
$$

The point lies exactly on the decision boundary.

---

### Case 4: $\xi_i > 1$

For example,

$$
\xi_i = 1.6.
$$

Then

$$
y_i(w^T x_i + b) \ge -0.6.
$$

Now the point may lie on the wrong side of the hyperplane.

- Margin violated ✅
- Misclassified ✅

---

## Step 5: Define the Optimization Objective

Like Hard-Margin SVM, we still want a **large margin**.

A large margin means minimizing

$$
w^T w.
$$

But now we also want the total margin violations to be as small as possible.

The total violation is

$$
\sum_{i=1}^{m}\xi_i.
$$

Therefore, the optimization problem becomes

$$
\boxed{
\min_{w,b,\xi}
\;
w^T w
+
C\sum_{i=1}^{m}\xi_i
}
$$

subject to

$$
\boxed{
y_i(w^T x_i + b)\ge1-\xi_i
}
$$

and

$$
\boxed{
\xi_i\ge0.
}
$$

This is called the **constrained optimization problem** because we minimize an objective while satisfying constraints.

---

## Step 6: What Does $C$ Do?

The parameter

$$
C > 0
$$

controls the trade-off between

- maximizing the margin.
- minimizing violations.

### Large $C$

A large value of $C$ gives a large penalty for every violation.

Therefore, SVM tries very hard to classify every training point correctly.

Usually this gives

- fewer violations.
- smaller margin.
- higher chance of overfitting.

### Small $C$

A small value of $C$ gives only a small penalty for violations.

Now SVM is willing to ignore a few difficult points if doing so creates a wider margin.

Usually this gives

- more violations.
- wider margin.
- better generalization.

---

## Step 7: Can We Remove the Constraints?

Until now, we have written the optimization problem with explicit constraints.

But solving constrained optimization is usually harder.

So we ask an important question:

> Can we write everything without explicitly writing the constraints?

The answer is **yes**.

---

## Step 8: Express the Slack Variable Directly

From the constraint

$$
y_i(w^T x_i+b)\ge1-\xi_i,
$$

move terms around

$$
\xi_i\ge1-y_i(w^T x_i+b).
$$

Also remember

$$
\xi_i\ge0.
$$

Therefore, for every training point, the slack variable must be the **larger** of these two values:

- $1-y_i(w^T x_i+b)$
- $0$

Hence

$$
\boxed{
\xi_i=
\max\left(1-y_i(w^T x_i+b),\,0\right).
}
$$

This is exactly the equation shown on your slide.

It is also commonly written as

$$
\xi_i=
\begin{cases}
1-y_i(w^T x_i+b), & \text{if } y_i(w^T x_i+b)<1,\\\\
0, & \text{if } y_i(w^T x_i+b)\ge1.
\end{cases}
$$

Both forms mean exactly the same thing.

---

## Step 9: Why Does This Formula Make Sense?

### Example 1

Suppose

$$
y_i(w^T x_i+b)=2.
$$

Then

$$
1-2=-1.
$$

So

$$
\xi_i=\max(-1,0)=0.
$$

No violation.

---

### Example 2

Suppose

$$
y_i(w^T x_i+b)=0.8.
$$

Then

$$
1-0.8=0.2.
$$

So

$$
\xi_i=0.2.
$$

Small margin violation.

---

### Example 3

Suppose

$$
y_i(w^T x_i+b)=-1.
$$

Then

$$
1-(-1)=2.
$$

So

$$
\xi_i=2.
$$

Large violation and likely misclassification.

---

## Step 10: Remove the Slack Variables

Earlier, we minimized

$$
w^T w
+
C\sum_{i=1}^{m}\xi_i.
$$

Now replace every

$$
\xi_i
$$

with

$$
\max(1-y_i(w^T x_i+b),0).
$$

We obtain

$$
\boxed{
\min_{w,b}
\;
w^T w
+
C
\sum_{i=1}^{m}
\max(1-y_i(w^T x_i+b),0)
}
$$

Notice that the slack variables have completely disappeared.

Everything now depends only on

- $w$
- $b$

This is called the **unconstrained formulation** because we no longer explicitly optimize the slack variables or write the constraints. They are automatically enforced through the loss function.

---

## Step 11: What is Hinge Loss?

The term

$$
\boxed{
\max(1-y_i(w^T x_i+b),0)
}
$$

is called the **hinge loss**.

It measures how much each training point violates the desired margin.

If

$$
y_i(w^T x_i+b)\ge1,
$$

then

$$
\text{Hinge Loss}=0.
$$

There is no penalty.

If

$$
y_i(w^T x_i+b)<1,
$$

then

$$
\text{Hinge Loss}=1-y_i(w^T x_i+b).
$$

The farther the point is inside the margin (or on the wrong side of the boundary), the larger the penalty.

---

## Step 12: Final Intuition

The final Soft-Margin SVM objective is

$$
\boxed{
\min_{w,b}
\;
w^T w
+
C
\sum_{i=1}^{m}
\max(1-y_i(w^T x_i+b),0)
}
$$

This objective has **two parts**.

### First Term

$$
w^T w
$$

tries to make the margin as large as possible.

### Second Term

$$
C
\sum_{i=1}^{m}
\max(1-y_i(w^T x_i+b),0)
$$

penalizes margin violations and misclassifications.

The parameter

$$
C
$$

decides how much importance is given to these violations.

- **Large $C$** → prioritize fewer violations (fit the training data more closely).
- **Small $C$** → prioritize a larger margin even if some violations are allowed.

---

## Connection with Logistic Regression

Finally, notice the similarity with **Logistic Regression**.

Both optimize the parameters $w$ and $b$ using gradient-based optimization methods.

The key difference is the loss function:

- **Logistic Regression** uses **Log Loss (Cross-Entropy Loss)**.
- **Soft-Margin SVM** uses **Hinge Loss**.

This is exactly what the last bullet on your slide is highlighting.
