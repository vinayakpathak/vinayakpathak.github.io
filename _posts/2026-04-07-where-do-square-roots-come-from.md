---
layout: post
title: "Where do square roots come from?"
date: 2026-04-09 17:06:56
description:
tags:
categories:
giscus_comments: true
---

At some point I used to do computational/combinatorial geometry and I'd have square roots appear in my calculations a lot. Most of the time they appeared due to the Pythagoras theorem. For example, the long diagonal of the $$n$$-dimensional cube circumscribing the unit ball, has length $$2\sqrt{n}$$.

Now I do learning theory, and square roots appear a lot still. And now they appear in convergence bounds. For example, with 99% probability, the empirical mean approximates the true mean of a (finite variance) random variable to within an additive error $$O(1/\sqrt{n})$$. Something I recently realized is that both square roots appear for almost the same reason. They are both consequences of things being an inner product space.

An [inner product space](https://en.wikipedia.org/wiki/Inner_product_space) is a vector space together with an inner product defined on pairs of vectors, denoted $$\langle u, v \rangle$$. Once you have inner products, you also get norms by taking the inner product of a vector by itself. That is, $$\lVert u \rVert^2 = \langle u, u \rangle$$. Once you have norms, square roots start to sound like things you might frequently encounter. Indeed, if inner product is norm squared, then norm is the square root of the inner product. However, the story becomes interesting only once you start calculating norms of _sums_ of vectors.

An inner product also satisfies linearity. In particular,

\begin{equation}
\langle u+v, w \rangle = \langle u, w \rangle + \langle v, w \rangle
\end{equation}

This, combined with symmetry, gives us a way to calculate the norm of sums of vectors. Thus, $$\lVert u + v \rVert^2 = \langle u+v, u+v\rangle = \lVert u \rVert^2 + \lVert v \rVert^2 + 2\langle u, v\rangle$$, and in particular, when $$\langle u, v \rangle = 0$$, we can simply write,

\begin{equation}\label{eq:pythagoras}\lVert u + v \rVert^2 = \lVert u \rVert^2 + \lVert v \rVert^2\end{equation}

Now if we consider the inner product space defined by vectors in a Euclidean space, then this gives us exactly the Pythagoras theorem[^pythagoras]. We don't need to stop at two vectors. We can add $$n$$ orthogonal vectors, each of unit length, and what we end up with will have a length $$\sqrt{n}$$. This is precisely what gives us the length of the diagonal of a cube in n-dimensions.

Why do we get $$\sqrt{n}$$ in convergence bounds?

Another interesting inner product space one can consider is the space of random variables. For any two random variables $$X$$ and $$Y$$, define their inner product as $$\langle X, Y \rangle = \mathbb{E}[XY]$$. One can check that this indeed forms an inner product space.

Now consider the task of approximating the true mean of a random variable $$X$$ using iid samples $$X_1,\ldots , X_n$$ from it. Consider the difference between the empirical mean and the true mean, i.e., $$\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}X$$. Write this as $$\frac{1}{n}\sum_{i=1}^n\left(X_i - \mathbb{E}X \right) = \frac{1}{n}\sum_{i=1}^n Z_i$$ where $$Z_i = X_i - \mathbb{E}X$$. Now think of $$Z_i$$'s as vectors in the inner product space of random variables and note that $$\langle Z_i, Z_j \rangle = \mathbb{E}[Z_iZ_j] = \mathbb{E}[(X_i-\mathbb{E}X)(X_j-\mathbb{E}X)] = \text{Cov}(X_i, X_j)$$. Also, $$\lVert Z_i \rVert^2 = \text{Var}(X_i)$$. But since the $$X_i$$'s are independent, the covariance is zero, and that means $$\langle Z_i, Z_j \rangle = 0$$. Thus $$Z_i, Z_j$$ are "orthogonal", and so $$\frac{1}{n}\sum_{i=1}^n Z_i = \frac{1}{n}\sqrt{\sum_{i=1}^n \lVert Z_i \rVert^2} = \sqrt{\frac{\text{Var}(X)}{n}} = O\left(\sqrt{\frac{1}{n}}\right)$$.

Once you can bound the variance of a random variable, you can prove tail bounds using [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality). This gives us the kinds of convergence bounds I talked about in the beginning.

So there we have it. Two seemingly different things are actually the same once we look at them using the right abstractions.

[^pythagoras]: Does this mean Pythagoras theorem is trivial once you have discovered inner product spaces? It is quite true that if we think of line segments in 2d as 2-dimensional vectors, and define the inner product in the usual linear algebra way, and define norm-squared as the inner product with itself, then equation $$\eqref{eq:pythagoras}$$ holds. However, the content of Pythagoras theorem is to show that what we think of as a "right angle" is the same thing as the inner product being zero, and what we think of as "length" is the same thing as the norm.
