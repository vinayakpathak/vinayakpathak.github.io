---
layout: post
title: "Faster than O(1/\\sqrt{n}) rate for agnostic learning?"
date: 2026-04-02 22:39:44
description:
tags:
categories:
giscus_comments: true
---

The story of binary classification (and many other learning problems) is that in the realizable case, we get a learning rate of $$O(1/n)$$, however for agnostic learning it's not possible to get anything better than $$\Omega(1/\sqrt{n})$$. Right? Right?

Ok, here's a learning problem where we can get $$O(1/n)$$ in the agnostic case. I don't know if there is a characterization for exactly when this is possible, but here is the problem.

Suppose your data distribution $$D$$ over $$X \times Y$$ is such that the marginal over $$Y$$ is uniform over $$\{-1, 1\}$$. Let the hypothesis class be the set of constant functions
$$
\mathcal{H} = \{h_c : X \to [-1, 1] \mid h_c(x) = c \text{ for all } x \in X,\ c \in [-1,1]\}.
$$
Let the loss be the quadratic loss, namely $$\ell(h, x, y) = (h(x) - y)^2$$. If the training sample is $$S = ((x_1, y_1), \ldots, (x_n, y_n)) \sim D^n$$ and we write
$$
\bar{Y} = \frac{1}{n}\sum_{i=1}^n y_i,
$$
then ERM returns the constant predictor $$h_{\bar{Y}}$$.

On the other hand, the optimal hypothesis for the population risk is clearly $$h_0$$, and its risk is $$R(0)=1$$. In particular, this is genuinely agnostic since the optimum risk is not zero.

The risk of ERM is
$$
R(\bar{Y}) = \mathbb{E}_{(x,y)\sim D}\left[(\bar{Y}-y)^2\right]
= \bar{Y}^2 - 2\bar{Y}\mathbb{E}[Y] + \mathbb{E}[Y^2]
= \bar{Y}^2 + 1,
$$
where we used the facts that $$\mathbb{E}[Y]=0$$ and $$\mathbb{E}[Y^2]=1$$.

And therefore, the excess risk is
$$
R(\bar{Y}) - R(0) = (\bar{Y}^2 + 1) - 1 = \bar{Y}^2.
$$
Taking expectation over the randomness of the training set,
$$
\mathbb{E}_S[R(\bar{Y}) - R(0)] = \mathbb{E}[\bar{Y}^2].
$$
But $$\bar{Y}$$ is the average of i.i.d. random variables with mean $$0$$ and variance $$1$$, hence
$$
\mathbb{E}[\bar{Y}^2] = \mathrm{Var}(\bar{Y}) = \frac{1}{n}.
$$
So the expected excess risk of ERM is exactly
$$
\mathbb{E}_S[R(\hat{h}_{\mathrm{ERM}}) - \inf_{h\in\mathcal{H}} R(h)] = \frac{1}{n}.
$$

Thus this agnostic learning problem has rate $$O(1/n)$$.
