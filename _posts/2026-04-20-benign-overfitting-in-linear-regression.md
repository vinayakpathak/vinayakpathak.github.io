---
layout: post
title: "Benign overfitting in linear regression"
date: 2026-04-20 22:24:50
description:
tags:
categories:
giscus_comments: true
---

This blog post is based on [this paper](https://arxiv.org/abs/1906.11300), in particular, its exposition in [this survey](https://arxiv.org/abs/2103.09177).

The story of machine learning theory is that a long long time ago we thought uniform convergence was crucial for generalization. That is, if you had a hypothesis class (for example, the set of all neural networks with a certain specific architecture), and you picked one hypothesis from that class based on the training set, then the main tool we had to show that its _test loss_ was small was to show that: a) its _training loss_ was small, and b) the hypothesis class had the property that with high probability over the sampling of the training set, every hypothesis in that class had a small difference between its test loss and train loss. Or in other words, proving generalization relied on proving that $$\lvert \hat{L}(h) - L(h) \rvert \leq \epsilon$$ with high probability. 

But then came along the "rethinking" [paper](https://arxiv.org/abs/1611.03530), and threw a wrench in the plan. It showed that you could train a neural network on _noisy_ data, i.e., data where you intentionally corrupt the labels of a $$p$$ fraction of the data points, and the network ended up perfectly fitting the training set and still getting the optimal loss on the test set. That is, you had $$\hat{L}(h) = 0$$ and $$L(h) = p/2$$ (assuming it's binary classification, it gets error 1/2 on $p$ fraction of the points). Thus a good test error was being achieved despite the previous strategy failing. Thus it was suggested that we needed some rethinking.

Achieving $$\hat{L}(h) = 0$$ on noisy data is pretty much the definition of overfitting. And yet, as the experiments in the "rethinking" paper showed, the network was not overfitting. This phenomenon was later named _benign overfitting_. 

It is actually not very hard to imagine scenarios where benign overfitting can happen. Let's go back to uniform convergence and pick a hypothesis class where uniform convergence does happen. Thus, given a training set, you pick a hypothesis from the class that minimizes the training error. Since the data here is noisy, you won't get a small training error. However, as long as your sample size is large enough, you do have the guarantee that the test error is close to the training error, and hence close to the optimum test error. But how do we find a hypothesis that does optimally on the test set while achieving _zero_ loss on the training set? This is quite simple. Suppose $$\hat{h}$$ is the result of minimizing the training loss. Consider a hypothesis $$h'$$ such that for any $$(x,y)$$ that's actually in the training set, $$h'(x) = y$$, and for everything else $$h(x) = \hat{h}(x)$$. For non-discrete problems, the training set is of measure zero, and therefore $$h'$$ has the same test loss as $$\hat{h}$$ and yet zero loss on the training set.

So the mystery isn't that benign overfitting happens, but that benign overfitting happens without adding this extra hypothesis-modification step in the end. In reality, we don't train a network and then modify it to fit the training data perfectly. We simply train a network and the trained network exhibits this strange property. How does that happen?

[Bartlett et al](https://arxiv.org/abs/1906.11300) studied this phenomenon in the simplest setting: linear regression. Imagine you are given $$n$$ points $$x_1,\ldots , x_n \in \mathbb{R}^d$$ along with their $$y$$ values $$y_1, \ldots , y_n\in \mathbb{R}$$. Suppose that $$n < d$$, thus a zero-training-loss solution can be achieved. Consider the training algorithm that picks the minimum norm solution among this set, i.e., we pick

<div markdown="0">
\begin{equation}
\label{eq:interp}
\hat{\theta} = \arg\min_\theta \{\lVert\theta\rVert^2: \theta^Tx_i = y_i, \forall i\leq n\}
\end{equation}
</div>
 
Since this _interpolates_, i.e., achieves zero loss on the training set, it's unclear a priori how it would perform on test set. We can't really use the uniform convergence strategy here. However, we can do some linear algebra magic and see that an interesting picture emerges.

Note, first of all, that the $$n < d$$ assumption is the culprit here. It is what's causing all this weirdness. For example, if $$n > \Omega(d)$$ satisfied whatever sample complexity requirements one gets from calculating the Rademacher complexity of the linear regression problem, then we would have that training loss and test loss were close to each other, and uniform convergence would perfectly explain the generalization. So let's just consider a lower dimensional part of the problem for the time being. Truncate the $$x$$'s to their first $$k$$ coordinates. Let $$X$$ be the original $$n\times d$$ _design matrix_ and let $$X_{\leq k}$$ be the $$n\times k$$ design matrix you get by keeping only the first $$k$$ coordinates, and $$X_{>k}$$ be the remaining coordinates. Now consider the following ridge regression problem in $$\mathbb{R}^k$$:

<div markdown="0">
\begin{equation}
\label{eq:ridge}
\hat{\theta}_{\leq k} = \arg\min_{\theta\in\mathbb{R}^k} \lVert X_{\leq k}\theta - y\rVert^2 + \gamma\lVert\theta\rVert^2
\end{equation}
</div>

One can show that under certain assumptions on $$X$$, the predictor that solves \eqref{eq:ridge} and outputs $$\hat{\theta}_{\leq k}^Tx_{\leq k}$$ for any $$x$$ produces very similar outputs as the predictor given by the solution to \eqref{eq:interp}. We will worry about the assumptions later, but if this is indeed true, we get that the interpolating solution is really just a lower dimensional ridge regression in disguise. Of course, a low-dimensional ridge regression guarantees uniform convergence but does not give us interpolation. This is where the remaining $$d-k$$ coordinates help. They provide enough leeway to fit the training points exactly without impacting the generalization loss by much. 

So in a sense this is similar to first learning using uniform convergence and then artificially changing the predictor so it fits the training points perfectly as described earlier. However, in this case, the artificial perturbation of the predictor is still a part of the original learning algorithm as opposed to an additional step.

Now, to see when and why \eqref{eq:interp} and \eqref{eq:ridge} are approximately equivalent, write \eqref{eq:interp} as:

<div markdown="0">
\begin{equation}
\label{eq:interp-split}
\hat{\theta} = \arg\min_{\theta_{\leq k}, \theta_{>k}} \{\lVert\theta_{\leq k}\rVert^2 + \lVert\theta_{>k}\rVert^2: X_{\leq k}\theta_{\leq k} + X_{>k}\theta_{>k} = y\}
\end{equation}
</div>

If we fix $$\theta_{\leq k}$$, then the problem becomes:

<div markdown="0">
\begin{equation}
\hat{\theta}_{>k} = \arg\min_{\theta_{>k}} \{\lVert\theta_{>k}\rVert^2: X_{>k}\theta_{>k} = y - X_{\leq k}\theta_{\leq k}\}
\end{equation}
</div>

This has the closed form solution which satisfies:

<div markdown="0">
\begin{equation}
\lVert\hat{\theta}_{>k}\rVert^2 = (y - X_{\leq k}\theta_{\leq k})^T(X_{>k}X_{>k}^T)^{-1}(y - X_{\leq k}\theta_{\leq k})
\end{equation}
</div>

Plugging this back into \eqref{eq:interp-split}, we get that:

<div markdown="0">
\begin{equation}
\hat{\theta}_{\leq k} = \arg\min_{\theta_{\leq k}} \{\lVert\theta_{\leq k}\rVert^2 + (y - X_{\leq k}\theta_{\leq k})^T(X_{>k}X_{>k}^T)^{-1}(y - X_{\leq k}\theta_{\leq k})\}
\end{equation}
</div>

If we could say that $$(y - X_{\leq k}\theta_{\leq k})^T(X_{>k}X_{>k}^T)^{-1}(y - X_{\leq k}\theta_{\leq k})\approx \lVert y - X_{\leq k}\theta_{\leq k}\rVert^2/\gamma$$, then we get exactly \eqref{eq:ridge}. But when can we say this? Basically we need $$X_{>k}X_{>k}^T$$ to sort of behave like a scaled $$I$$. The paper shows certain distributional assumptions for which this holds with high probability. It eventually boils down to the eigendecomposition of $$\mathbb{E}[xx^T]$$ and how flat its tail is. If the tail is flat enough, then one can show that the above holds once we write everything in the basis of the eigenvectors of $$\mathbb{E}[xx^T]$$.