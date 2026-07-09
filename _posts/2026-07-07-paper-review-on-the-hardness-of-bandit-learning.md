---
layout: post
title: "Paper Review: On the Hardness of Bandit Learning"
date: 2026-07-07 00:00:00
description:
tags:
categories:
giscus_comments: true
---

Figuring out the computational complexity of learning is an important research question. The theory of learning has so far managed to prove that [very simple things are computationally hard]({% post_url 2026-05-03-computational-complexity-of-binary-classification %}) to learn. And yet, neural networks are learning things from real world data [without having to solve NP-hard problems](https://arxiv.org/abs/quant-ph/0502072). So something is happening in the computational complexity department that we do not understand.

I came across [this paper](https://arxiv.org/abs/2506.14746) by Nataly Brukhim, Aldo Pacchiano, Miroslav Dudik, and Robert Schapire that attempts to understand the computational hardness of bandit problems. While many "general" theorems are known about computational complexity of PAC-learning, not a lot was known for bandits. This paper has an interesting result.

## What Computational Problem is "Bandit Learning"?

Recall that in a bandit problem, you have a set $$\mathcal{A}$$ of actions and a set $$\mathcal{F}$$ of functions. Any $$f \in \mathcal{F}$$ assigns an expected _reward_ to each action, i.e., $$f: \mathcal{A}\rightarrow [0,1]$$. An adversary picks some $$f^*\in\mathcal{F}$$ and a learner interacts with it as follows. At time step $$t$$, the learner picks action $$a_t \in \mathcal{A}$$ and observes a reward $$r_t$$ satisfying $$\mathbb{E}[r_t\mid a_t]=f^*(a_t)$$. The learner does not know $$f^*$$ but does know $$\mathcal{F}$$. The goal of the learner is to minimize the pseudo-regret:

$$
R(T) = T\max_{a\in\mathcal{A}} f^*(a) - \sum_{t=1}^T f^*(a_t)
$$

We typically say that learning has been successful if there exists a way of picking the $$a_t$$'s so that $$R(T)$$ is sublinear, i.e., $$\lim_{T\rightarrow\infty}\frac{R(T)}{T} = 0$$. But now we want to understand the computational complexity, so we need to be more careful in defining the exact computational problem we intend to solve. In particular, what is the input and what is the output? And when do we say that a learner is "computationally efficient"?

At each timestep $$t$$, the learner needs to produce an action $$a\in\mathcal{A}$$. This means we need an encoding of actions so the learner can write out the action it wants to take. For simplicity, we can assume that $$\mathcal{A}$$ is the $$n$$-dimensional boolean hypercube $$\{0,1\}^n$$ and thus an action is simply a boolean string of length $$n$$. For now, this is enough to define the computational problem. At each time step $$t$$, the learner gets to see the history of actions and rewards and needs to output $$a_t\in\mathcal{A}$$. Thus it is reasonable to say that the learner is computationally efficient if it takes time and space $$\operatorname{poly}(n, t)$$ at timestep $$t$$. This rules out algorithms that scan all $$2^n$$ possible actions before deciding on a single action.

What about the regret? We have already agreed that we are going to call a bandit problem learnable if we can get regret to be sublinear in $$T$$. But here we are asking when is a bandit problem _efficiently_ learnable? This leads to two questions: 1) should we be satisfied by a mere sublinear dependence of regret on $$T$$ or should we ask for something faster? 2) how should the regret depend on $$n$$?

Suppose we say that regret should depend polynomially on $$n$$ and sublinearly on $$T$$, i.e., $$R(T) = \operatorname{poly}(n)o(T)$$. Is this a good definition of computationally and statistically efficient bandit learning?

I would say that this doesn't feel very efficient. Moreover, this particular guarantee is somewhat trivial to obtain. We can basically brute force over the entire action space $$\mathcal{A}$$ to find the best arm and then play it until eternity. Brute forcing can be done by playing one arm at a time, and thus requires only polynomial time per time step. We do need to maintain the best arm, and that can also be done with a polynomial amount of space. But why does this satisfy the regret guarantee of $$\operatorname{poly}(n)o(T)$$? Shouldn't the dependence on $$n$$ be exponential if we brute force? The reason is that if the dependence on $$T$$ is allowed to grow arbitrarily slowly then we can hide some dependence on $$n$$ within it. To see this, assume for now that we need to query each arm exactly once in order to identify the best arm. In general, this is not true, and the number of times we need to query each arm is a constant that depends on the noise. But for simplicity, we need to make $$2^n$$ queries in total before we know the best arm. The strategy above gives a regret $$R(T) = \min(T, 2^n)$$. The claim is that $$\min(T, 2^n) \leq O(n)\cdot\frac{T}{\log T}$$. This can be easily verified by considering the two cases where $$T \leq 2^n$$ and $$T > 2^n$$ and substituting.

Then what is a good definition of computationally and statistically efficient bandit learning?

## Polynomial Regret vs Best Arm Identification

Another option is to be a bit more quantitative about exactly how quickly we want the average regret to go to zero. Earlier we were satisfied with a mere sublinear regret. But what if we require that $$R(T) \leq \operatorname{poly}(n)T^c$$ for some fixed $$c < 1$$? Let's call this problem bandits with polynomial regret. After all, we do want statistical efficiency, and polynomial regret sounds a lot more efficient than a mere sublinear regret. 

For a fixed $$c < 1$$, another way to interpret this problem is to say that the learner is also given an $$\epsilon < 1$$ as input and needs to ensure that $$R(T)/T \leq \epsilon$$ after a $$poly(n, 1/\epsilon)$$ amount of total computation. This is only possible if it runs in polynomial time per timestep and achieves a polynomial regret. 

The paper mostly considers a slightly different bandit problem, called the best arm identification problem. Here, we do not necessarily want a low regret, but merely want to identify an arm that is $$\epsilon$$-close to the optimal arm. That is, we want to find $$\hat{a}\in\mathcal{A}$$ such that $$f^*(\hat{a}) \geq \max_{a\in\mathcal{A}} f^*(a) - \epsilon$$. The $$(\epsilon, \delta)$$-BAI problem is defined as follows.

<div class="definition">
  <p><strong>Definition (\((\epsilon,\delta)\)-best-arm identification).</strong> Let \(\mathcal{A}\) be an action set and let \(\mathcal{F}\subseteq [0,1]^{\mathcal{A}}\) be a known class of reward functions. Nature chooses an unknown target function \(f^*\in\mathcal{F}\). A learner is given query access to \(f^*\), i.e., at each round \(t\), it chooses an action \(a_t\in\mathcal{A}\) and observes a reward \(r_t\) satisfying \(\mathbb{E}[r_t\mid a_t]=f^*(a_t)\).</p>

  <p>The learner solves the \((\epsilon,\delta)\)-BAI problem for \(\mathcal{F}\) with query complexity \(m(\epsilon,\delta)\) if, after at most \(m(\epsilon,\delta)\) queries, it outputs an action \(\hat{a}\in\mathcal{A}\) such that, for every \(f^*\in\mathcal{F}\), with probability at least \(1-\delta\),</p>

  \[
f^*(\hat{a}) \geq \sup_{a\in\mathcal{A}} f^*(a) - \epsilon.
  \]

  <p>We say that the learner is efficient if \(m(\epsilon, \delta)\), as well as its total running time, is \(\operatorname{poly}(n, 1/\epsilon, 1/\delta)\), where \(n\) is the representation length of the action set and function class.</p>
</div>

One can show that the polynomial regret problem defined above and the efficient BAI problem are equivalent. Indeed, if you have a learner for BAI, just run it to identify the best arm and then play that arm for the rest of the time steps. This gives us polynomial regret. On the other hand, if you have a polynomial regret learner, then run it for $$T$$ rounds, where $$T$$ will depend on $$\epsilon$$ and $$\delta$$ and pick a random arm uniformly among the $$T$$ arms played. Using Markov inequality one can show this to be an $$\epsilon$$-optimal arm with high probability.

In conclusion, the existence of an efficient $$(\epsilon, \delta)$$-BAI learner is an excellent candidate for what one might call efficient bandit learnability.

## When is Efficient BAI Doable?

So now, just like the paper, we can focus on the BAI problem. Right away, we can realize that without any structure on $$\mathcal{F}$$, nothing can be done. For example, suppose $$\mathcal{F}$$ is the set of all singletons, i.e., for each $$a\in\mathcal{A}$$, $$\mathcal{F}$$ contains an $$f_a$$ such that $$f_a(a) = 1$$ and $$f_a(a') = 0$$ for $$a' \neq a$$. Nature has picked one of these functions and the learner does not know which one. In order to figure out the best arm, the learner needs to find the arm which gets the value of one by this function. This can only be done by querying all $$2^n$$ arms and thus efficient BAI is impossible. This means it is important that at least the number of queries required for BAI should be polynomial. But is that sufficient? That is, can we think of a function class $$\mathcal{F}$$ for which the number of queries required is small, and yet the amount of computation required is large?

This is also not hard to construct. The main hardness theorem in the paper is stated in the noiseless setting, where rewards are deterministic functions of the arm. The authors later discuss how analogous noisy versions can be obtained, but we will stick with the noiseless setting here. Let $$\mathcal{F}$$ be the set of all polynomial sized Boolean formulas. To calculate $$f(a)$$ we just evaluate the formula on the Boolean string corresponding to $$a$$. Now consider that we are given an additional arm $$\star$$ such that the reward for that arm is a real number encoding the formula. A polynomially-sized Boolean formula can be encoded with polynomially many bits, and we can arrange those bits in a way that the rewards lie within a small range, say $$[1/4,1/3]$$. Now the learner can query the arm $$\star$$ once and infer the exact hidden $$f^*$$. But to find the best arm, it needs to find an assignment to the $$n$$ variables such that the formula evaluates to one. This is NP-hard. Thus even though BAI requires only one query, it is still _computationally_ intractable.

One might object that these are fairly unnatural instances of the bandit problem and thus it's not that surprising that they are hard. The paper goes one step further and proves hardness under even stronger assumptions.

One strategy for doing bandit learning or BAI could be something like the following:
1. Try a bunch of arms and observe the rewards.
2. Find a function in the function class that is consistent with those rewards.
3. Assume that that is the true function and find the best arm for that function.

To implement this strategy efficiently, we need to be able to solve #2 and #3 efficiently. The paper presents an example where both of them can be done efficiently and yet BAI is computationally intractable.

The construction is quite neat. The example builds on top of the previous example. We now consider three kinds of arms. The first kind $$\mathcal{A}_1$$ just contains the arm $$\star$$. The second kind $$\mathcal{A}_2$$ contains all Boolean strings of length $$n$$. And the third kind $$\mathcal{A}_3$$ contains $$2^n$$ arms numbered from 1 to $$2^n$$. 

Note that so far we were ignoring the encoding for the functions because the learner did not really need to output a function. However, solving #2 requires the algorithm to output a function and thus we need to specify how each function in the function class is represented using a finite encoding. 

The function class now is going to have two kinds of functions. First is the set of all polynomially sized Boolean formulas (actually the paper uses quadratically-sized for some technical reasons) and thus can be indexed by formula $$\phi$$. The second set consists of functions indexed by the pair $$\phi, c$$ where $$c \in [1..2^n]$$ and $$\phi$$ is some _satisfiable_ formula. 

As before, all of the functions use the arm $$\star$$ to encode their corresponding Boolean formula into a real number. For function $$f_{\phi, c}$$ of the second kind, it assigns $$c \in \mathcal{A}_3$$ the value 1, and something smaller to everything else. This means $$c$$ is the best arm for $$f_{\phi, c}$$ and thus can be easily read from the function's encoding. Thus #3 is easily solved. We still need to specify what $$f_{\phi, c}$$ assigns to all the other arms. For an arm in $$\mathcal{A}_3$$ that's not $$c$$, $$f_{\phi, c}$$ assigns it zero. For arms in $$\mathcal{A}_2$$, if it is a satisfying assignment to $$\phi$$, then the function gives it a value $$c/2^{n+1}$$, and otherwise 0. Thus to actually find the $$c$$ using only queries, one needs to find a satisfying assignment to $$\phi$$. 

Finally, we can also see that #2 is tractable here. Given a sequence of (action, reward) pairs, we need to find a function consistent with them. If any of the arms happen to be $$\star$$, then this is easy. Otherwise, we need to be able to find Boolean formulas that fit a set of given values. This is also easy.

Taking a step back, what does this hardness show? It appears that the three step strategy outlined above is not enough for bandit learning. In particular, we need a tractable way to _explore_. That is, it is not enough to be able to "try a bunch of arms". We need to be clever about exactly which arms to try and sometimes this cleverness might not be computationally tractable.
