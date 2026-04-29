---
layout: post
title: "Paper Review: How Does Machine Learning Manage Complexity?"
date: 2026-04-28 00:00:00
description:
tags:
categories:
giscus_comments: true
---

I recently came across this interesting [paper](https://arxiv.org/pdf/2604.07233) by Lance Fortnow titled "How Does Machine Learning Manage Complexity?" I am very happy to see complexity theorists getting into some ML theory. If nothing else, I am looking forward to seeing more papers that start by saying:

> In this paper all distributions will be over $$\{0, 1\}^n$$.

Instead of "let $$\mathcal{X}$$ be a Polish space equipped with its Borel $$\sigma$$-algebra."

The paper makes some interesting observations at the intersection of ML theory and complexity theory and here are two I want to make a note of for the future.

#### What complexity class does a language model belong to?

Assume that language models operate on a vocabulary of size two, i.e., $$\{0,1\}$$. Consider a language model that takes a sequence of length $$n$$ as input and produces the probability that the next token is a one. What complexity class is this computation in? Of course, since it does successfully happen in the real world, it must be "efficient". But is it in P? 

The paper makes the argument that it's better to think of it as a polynomial-sized circuit with non-uniformity allowed aka P/poly. The neural network architecture kind of looks like a polynomial-sized circuit so it makes sense to think of it as a circuit. But why should we allow it to be non-uniform? The argument is that the training process is long and arduous and so we don't want to make an assumption that that is also efficient, certainly not efficient in terms of the input size $$n$$. The training process finds the weights and thus creates the circuit. We can think of the training process as the thing that finds the polynomial-sized advice.

#### How are neural networks able to represent real-world distributions?

The paper points out a paradox. Real-world distributions are efficiently sampleable. So let's say they are P-sampleable, i.e., there exists a randomized algorithm that runs in time polynomial in $$n$$ and samples a binary string of length $$n$$ from the real-world distribution. However, the distribution over n-bit strings represented[^nn] by a neural network is P/poly-computable as opposed to merely sampleable. This is because given any binary string, one can compute the probability assigned to it by the neural network[^teacher-forcing], and we are assuming that the inference algorithm is in P/poly. But we know that there are many P-sampleable distributions that are not P/poly-computable. So how does a neural network represent them?

First, what are examples of P-sampleable distributions that are not P/poly-computable? 

Consider, for example, a distribution over $$2n$$-bit strings where the last $$n$$ bits $$x$$ come from a P-sampleable distribution, and the first $$n$$ bits are $$f(x)$$ where $$f$$ is a one-way function. This is clearly P-sampleable. However, if it were also P-computable, then we would be able to invert $$f$$. Indeed, given $$f(x)$$, we could construct $$x$$ one bit at a time and check if the concatenation of $$f(x)$$ and the partially constructed $$x$$ has a non-zero probability. Thus first, we check the probability of $$f(x)+1$$ and $$f(x)+0$$ (where $$+$$ denotes concatenation), and if $$f(x)+1$$ is non-zero, we check $$f(x)+11$$ and $$f(x)+10$$, and so on.

Another fairly natural example comes from pseudo-random generators (PRG). A PRG for n-bit strings is an efficiently computable function $$f: \{0,1\}^l \rightarrow \{0,1\}^n$$ for some $$l < n$$ that takes a (uniformly) randomly generated $$l$$-bit "seed" and produces an $$n$$-bit string that looks as good as uniformly random to every efficient algorithm. Let $$\text{PRG}$$ denote the distribution induced on $$\{0,1\}^n$$ by a given pseudo-random generator, and let $$U$$ be the uniform distribution over $$\{0,1\}^n$$. Then it is a successful PRG if for any (randomized) P/poly algorithm $$A$$, and for every $$k>0$$, we have that:

\begin{equation}
\label{eq:prg}
\left|\Pr_{x \sim \text{PRG}}\left[A(x)=1\right] - \Pr_{x \sim U}\left[A(x)=1\right]\right| \leq \frac{1}{n^k}.
\end{equation}

It is standard to define $$\operatorname{negl}(n)$$ as a function such that for every $$k>0$$, $$\operatorname{negl}(n) \leq \frac{1}{n^k}$$. Thus one could write the above equation with the right hand side replaced by $$\operatorname{negl}(n)$$.

Clearly, a PRG is P-sampleable. Suppose it was also P/poly-computable. Then consider an algorithm $$A$$ that takes $$n$$-bit strings as inputs such that $$A(x) = 1$$ iff $$\Pr_\text{PRG}[x] > 0$$. Thus $$\Pr_{x \sim \text{PRG}}\left[A(x)=1\right] = 1$$, but $$\Pr_{x \sim U}\left[A(x)=1\right] \leq 2^{-(n-l)}$$. This clearly violates \eqref{eq:prg}. Thus the distribution could not have been P/poly-computable. 

Now that we have seen some examples of P-sampleable distributions that are not P/poly-computable, let's get back to the paradox. How does the paper resolve this? The resolution is not that neural networks can learn a P/poly-computable distribution that is indistinguishable from the real one if such a distribution exists. The paper formalizes this using the following (paraphrased) theorem:

**Theorem.** Let $$D$$ be a P-sampleable distribution over $$\{0,1\}^n$$ and let $$\mu$$ be a distribution learned by minimizing $$\operatorname{KL}(D \mathrel{\Vert} \mu)$$. Suppose there exists a P/poly-computable distribution $$\nu$$ such that $$\nu$$ is indistinguishable from $$D$$ in the sense of \eqref{eq:prg}. Then $$\mu$$ is very close to $$\nu$$ in KL-divergence, i.e., $$\operatorname{KL}(\nu \mathrel{\Vert} \mu)<\operatorname{negl}(n)$$. In particular, $$\mu$$ is also indistinguishable from $$D$$.

Thus, for example, if $$D$$ is the distribution generated by a PRG, then the model ends up learning something that's indistinguishable from a uniform distribution. 

**Proof.** Let's prove this for the case when both $$\nu$$ and $$\mu$$ have bounded max-entropy, i.e., there exists a $$k>0$$ such that $$\log\frac{1}{\mu(x)}\leq n^k$$ and $$\log\frac{1}{\nu(x)}\leq n^k$$ for all $$x$$. The paper extends it to unbounded max-entropy as well.

Define $$f(x) = \log\frac{\nu(x)}{\mu(x)}$$. Note that $$f(x)$$ is P/poly-computable. Also, $$\lvert f(x)\rvert\leq n^k$$, because both $$\log\frac{1}{\mu(x)}$$ and $$\log\frac{1}{\nu(x)}$$ are between $$0$$ and $$n^k$$. This means $$\lvert\mathbb{E}_{x\sim D} f(x) - \mathbb{E}_{x\sim \nu} f(x)\rvert\leq\operatorname{negl}(n)$$ because otherwise the process $$A(x)$$ that outputs 1 with probability $$\frac{f(x)+n^k}{2n^k}$$ would be able to distinguish between $$D$$ and $$\nu$$. Also, note that $$\mathbb{E}_{x\sim D} f(x) = \operatorname{KL}(D \mathrel{\Vert} \mu) - \operatorname{KL}(D \mathrel{\Vert} \nu)\leq 0$$, because $$\mu$$ was found by minimizing the KL with $$D$$. Thus $$\operatorname{KL}(\nu \mathrel{\Vert} \mu)=\mathbb{E}_{x\sim \nu} f(x)\leq\operatorname{negl}(n)$$.

Finally, by Pinsker's inequality, $$d_{\mathrm{TV}}(\nu,\mu)\leq \sqrt{\frac{1}{2}\operatorname{KL}(\nu \mathrel{\Vert} \mu)}$$. Since the square root of a negligible function is still negligible, $$\nu$$ and $$\mu$$ are negligibly close in total variation distance. Thus, for every efficient distinguisher $$A$$, the distinguishing gap between $$D$$ and $$\mu$$ is at most the distinguishing gap between $$D$$ and $$\nu$$ plus the total variation distance between $$\nu$$ and $$\mu$$, which is negligible.






[^nn]: A language model models $$p(y\mid x)$$ where $$x$$ is any $$k$$-bit string with $$k<n$$, and $$y\in\{0,1\}$$. This, together with a distribution on the first bit, induces a distribution over all $$n$$-bit strings. We say that this is the distribution represented by the language model.

[^teacher-forcing]: This is fairly straightforward for transformer models. For any string $$x$$, you can simply feed it to the model as input, and read off the probability in front of each bit. Multiplying all of them gives you the probability of the entire string.
