---
layout: post
title: "Computational Complexity of Binary Classification"
date: 2026-05-03 20:56:05
description:
tags:
categories:
giscus_comments: true
---

PAC learning for binary classification is a very well-studied problem and most variations in definitions turn out to be equivalent to each other if all we care about is sample complexity. Recall the following definitions:

Let $$\mathcal{X}_n$$ be an instance space, say $$\{0,1\}^n$$, and let $$H_n \subseteq \{0,1\}^{\mathcal{X}_n}$$ be a class of binary classifiers. A data distribution $$\mathcal{D}$$ is any distribution over $$\mathcal{X}_n\times \{0,1\}$$. For a classifier $$h:\mathcal{X}_n\rightarrow \{0,1\}$$, define its error as

<div markdown="0">
\begin{equation}
\operatorname{err}_{\mathcal{D}}(h) = \Pr_{(x,y)\sim \mathcal{D}}\left[h(x)\neq y\right].
\end{equation}
</div>

The best error achievable by the class $$H_n$$ under $$\mathcal{D}$$ is

<div markdown="0">
\begin{equation}
\operatorname{opt}_{\mathcal{D}}(H_n) = \inf_{h\in H_n} \operatorname{err}_{\mathcal{D}}(h).
\end{equation}
</div>

An agnostic PAC learner for $$H = \{H_n\}_{n\geq 1}$$ is a randomized algorithm $$A$$ such that for every $$n$$, every distribution $$\mathcal{D}$$ over $$\mathcal{X}_n\times \{0,1\}$$, and every $$\epsilon,\delta>0$$, if $$A$$ is given $$m(n,\epsilon,\delta)$$ independent samples from $$\mathcal{D}$$, then with probability at least $$1-\delta$$ it outputs a classifier $$h$$ satisfying

<div markdown="0">
\begin{equation}
\operatorname{err}_{\mathcal{D}}(h) \leq \operatorname{opt}_{\mathcal{D}}(H_n)+\epsilon.
\end{equation}
</div>

the learner is _sample-efficient_ if the number of samples it uses is polynomial in the natural parameters, i.e

<div markdown="0">
\begin{equation}
m(n,\epsilon,\delta) \leq \operatorname{poly}\left(n,\frac{1}{\epsilon},\frac{1}{\delta}\right).
\end{equation}
</div>

If $$\operatorname{opt}_{\mathcal{D}}(H_n) = 0$$, then we say that the problem is _realizable_.

If we ask that the learner outputs a hypothesis $$h\in H_n$$, then it's called _proper_ learning, otherwise it's _improper_ learning. 

Thus we have two axes: agnostic vs realizable, and proper vs improper. From standard results in learning theory, we know that sample-efficient learning in all four cases is characterized by the VC dimension of $$H_n$$. This also means they are all equivalent to each other.

However, things become a lot more messy when we study _computationally_-efficient learning. Here, we are not satisfied with a polynomial sample complexity, but also require that the learner runs in time polynomial in $$n, 1/\epsilon, 1/\delta$$. As all things computational, these learning problems are hard to characterize. Indeed, no characterizations are know. But perhaps one can still investigate the relative hardness of the four versions. It's obvious that agnostic cannot be easier than realizable (since realizable is a special case) and proper cannot be easier than improper (since a proper learner is also an improper learner). But which ones of these relationships are strict? Can we show, for example, that proper and improper are actually equivalent?

An old and beautiful result from Pitt and Valiant showed that in the realizable case, proper and improper are genuinely different. They gave an example of a learning problem that was computationally hard in the proper case (unless RP = NP), but efficient in the improper case. The problem is also a fairly natural one.

Let $$H_n$$ be the class of DNF formulas in $$n$$ variables with $$k$$ terms. That is, each hypothesis in $$H_n$$ is of the form $$T_1\lor\cdots\lor T_k$$ where each $$T_i$$ is a conjunction of literals. Pitt and Valiant showed that this class is hard to learn properly in the realizable case unless RP = NP. We will get to the proof later. First let's see how it can successfully be learned in the improper case. We begin by converting the DNF formula into its CNF form. This can be done by picking one literal from each $$T_i$$ and taking their disjunction to form one clause of the CNF. Then taking a conjunction of all clauses thus formed, we get the equivalent CNF formula[^dnf-to-cnf]. Since there are $$k$$-terms in the original DNF, we get that the CNF will have $$k$$ literals per clause. 

Interestingly, even though we associate CNF formulas (with more than two literals per clause) with various kinds of hardness results, when it comes to PAC learning, these things can be learned efficiently in the realizable setting. This is because the total number of clauses of size at most $$k$$ that one can create using $$n$$ variables is $$O(n^k)$$ and thus the total number of formulas one can create with these clauses is at most $$2^{O(n^k)}$$. This is a finite hypothesis class and thus has a VC dimension of at most $$O(n^k)$$ and thus running ERM on the training examples should successfully learn it. ERM can also be run in polynomial time. The idea is to go through the positive examples and keep exactly the clauses that are satisfied by all of them. The conjunction of all clauses still consistent with the positive examples constitutes the final CNF formula. Since the sample is realizable, every negative example must falsify at least one of these retained clauses, and hence the final CNF labels the whole sample correctly.

Ok, so k-CNF formulas can be efficiently and properly PAC-learned in the realizable setting. This sounds like a very large class of hypotheses. What goes wrong when we try to learn DNF formulas instead? One can show that a DNF formula with just two conjunctions $$T_1\lor T_2$$ is impossible to learn properly in the realizable setting unless $$RP=NP$$.

First note the following general result. Consider doing proper learning for a hypothesis class $$H$$ under the realizable setting. Consider also the computational problem: given labeled points $$(x_1, y_1),\ldots , (x_n, y_n)$$, decide if there exists some $$h\in H$$ such that $$h(x_i) = y_i$$ for all $$i$$. It is easy to see that if a computationally efficient proper PAC learner exists in the realizable setting then this decision problem is in RP. Indeed, let $$D$$ be the uniform distribution over $$(x_1, y_1), \ldots , (x_n, y_n)$$, and invoke the proper PAC learner with $$\epsilon < 1/n$$ and $$\delta = 1/3$$. If an $$h\in H$$ exists that correctly labels all the examples, then we are in the realizable setting, and the proper PAC learner returns a hypothesis of error less than $$1/n$$ with probability at least $$2/3$$. Under the uniform distribution over the sample points, this means it labels every point correctly. If no such $$h$$ exists, then no matter what the learner returns, the final verification step will reject it. Thus we can simply run the PAC learner and check whether the returned hypothesis labels all points correctly. If it does, then we say yes, otherwise we say no. In the yes case, we say yes with probability at least $$2/3$$, and in the no case we always say no. Thus we have found an algorithm in RP for the decision problem.

We can apply this argument to show hardness for proper PAC learning in the realizable case. For a given hypothesis class, we just need to prove that the decision problem of checking whether a hypothesis exists that correctly labels all input examples is hard. For the hypothesis class of length-2 DNF formulas, we can reduce an NP-complete problem to it, thus proving that efficient proper learning is possible only if RP = NP.

The reduction is from the set splitting problem, which is known to be NP-complete. An instance of the set splitting problem is a set system, i.e., a set $$S$$ and a set $$F$$ of its subsets, and the task is to decide whether $$S$$ can be split into two disjoint parts such that each subset in $$F$$ contains elements from each part. We may assume each set in $$F$$ has size at least two; set splitting remains NP-complete under this restriction. Given an instance of the set splitting problem, we construct the following labeled examples.

Let $$S=\{s_1,\ldots,s_n\}$$. For each element $$s_i$$, we have a variable $$x_i$$. For each $$s_i\in S$$, create the point $$p_i=(1,\ldots,1,0,1,\ldots,1)$$, where the unique zero is in coordinate $$i$$, and label it $$1$$. Next, for each $$f\in F$$, create the point $$q_f$$ where

<div markdown="0">
\begin{equation}
(q_f)_i =
\begin{cases}
0 & \text{if } s_i\in f,\\
1 & \text{otherwise}.
\end{cases}
\end{equation}
</div>

Label each $$q_f$$ by $$0$$. We now show that a valid partition of $$S$$ exists if and only if there is a 2-term DNF formula that correctly labels all these examples.

First suppose that $$S$$ has a valid split $$S=S_1\cup S_2$$. Define

<div markdown="0">
\begin{equation}
T_1=\bigwedge_{s_i\in S_2} x_i,
\qquad
T_2=\bigwedge_{s_i\in S_1} x_i,
\qquad
h=T_1\lor T_2.
\end{equation}
</div>

Consider a positive example $$p_i$$. If $$s_i\in S_1$$, then the unique zero of $$p_i$$ does not appear in $$T_1$$, so $$T_1(p_i)=1$$. Similarly, if $$s_i\in S_2$$, then $$T_2(p_i)=1$$. Thus $$h(p_i)=1$$ for every positive example.

Now consider a negative example $$q_f$$. Since the split is valid, $$f$$ contains some element of $$S_1$$ and some element of $$S_2$$. Therefore $$q_f$$ has a zero in a variable appearing in $$T_2$$ and a zero in a variable appearing in $$T_1$$. Hence both terms are false, so $$h(q_f)=0$$.

For the converse, suppose there is a 2-term DNF $$h=T_1\lor T_2$$ that correctly labels all the examples. We may assume without loss of generality that $$T_1$$ and $$T_2$$ contain only positive literals. Indeed, a term with two or more negated literals cannot satisfy any positive example $$p_i$$, because each $$p_i$$ has only one zero. Such a term can simply be deleted. A term with exactly one negated literal $$\lnot x_i$$ can only help on the positive example $$p_i$$; it can be replaced by the positive term $$\bigwedge_{j\neq i}x_j$$, which still accepts $$p_i$$ and rejects every negative example $$q_f$$, since every set $$f$$ in a nontrivial set-splitting instance has size at least two.

So write

<div markdown="0">
\begin{equation}
T_1=\bigwedge_{i\in A_1}x_i,
\qquad
T_2=\bigwedge_{i\in A_2}x_i.
\end{equation}
</div>

Since $$p_i$$ is labeled $$1$$, it must satisfy at least one of the two terms. Equivalently, for every $$i$$, either $$i\notin A_1$$ or $$i\notin A_2$$. We define a split as follows: put $$s_i$$ in $$S_1$$ if $$i\notin A_1$$, and put it in $$S_2$$ otherwise. This is well-defined because if $$i\in A_1$$, then the fact that $$p_i$$ is positive forces $$i\notin A_2$$.

It remains to check that every $$f\in F$$ is split. Since $$q_f$$ is labeled $$0$$, both terms must reject it. Since $$T_1(q_f)=0$$, there is some $$i\in A_1$$ with $$s_i\in f$$. By the definition of the split, this element lies in $$S_2$$. Similarly, since $$T_2(q_f)=0$$, there is some $$j\in A_2$$ with $$s_j\in f$$, and this element lies in $$S_1$$. Therefore $$f$$ intersects both $$S_1$$ and $$S_2$$.

Thus the examples are consistent with a 2-term DNF if and only if the original set-splitting instance is satisfiable. Since set splitting is NP-complete, the consistency problem for 2-term DNF is NP-hard. By the previous argument, an efficient proper PAC learner for 2-term DNF in the realizable setting would put this NP-hard decision problem in RP, and hence would imply $$\mathrm{RP}=\mathrm{NP}$$.

Ok, so now we have for the realizable case, an example where proper learning is hard but improper is easy. Thus being able to do improper realizable learning efficiently does not imply we can also do proper realizable learning. But what if we can do improper _agnostic_ learning efficiently? Can that imply efficient learner for proper realizable learning too? It is kind of a strange thing to do to compare agnostic improper against realizable proper, and thus I couldn't find a paper that explicitly studies this problem. Thus I deployed GPT 5.5 Pro on it and it found an amazingly elegant counterexample. It says it's not new and cites a few papers that have had "similar ideas". But looking at those papers, I don't find their ideas to be all that similar. So either GPT is being humble or it's just not citing the correct thing. But the counterexample is quite simple and it also subsumes the previous counterexample. Indeed, since we get an example where agnostic improper learning is easy but realizable proper learning is hard, the same example also proves that it's possible for realizable improper learning to be easy while realizable proper learning is hard.

Here is the counterexample. 

For each $$n$$, let $$X_n$$ be the set of all 3-CNF clauses over variables $$x_1,\ldots,x_n$$. Thus, an element of $$X_n$$ is a disjunction of three literals. Let's allow repeated literals just so that every clause has exactly three slots. Thus the domain is small, i.e., $$\lvert X_n\rvert \leq O(n^3)$$.

Now for every assignment $$a\in\{0,1\}^n$$, define a classifier $$h_a:X_n\rightarrow \{0,1\}$$ by

<div markdown="0">
\begin{equation}
h_a(C)=1
\quad\Longleftrightarrow\quad
a\text{ satisfies the clause }C.
\end{equation}
</div>

Let $$H_n=\{h_a:a\in\{0,1\}^n\}$$. We can sort of see why this might be hard to learn properly. Indeed, if there is an efficient proper PAC learner for $$H$$ in the realizable setting, then we can use that to solve 3-SAT using a similar reduction as earlier.

Take a 3-CNF formula $$\varphi = C_1\land C_2\land\cdots\land C_m$$.

Define a distribution $$D_\varphi$$ over labeled examples by choosing $$j\in[m]$$ uniformly at random and outputting $$(C_j,1)$$. If $$\varphi$$ is satisfiable, then there is an assignment $$a^\star$$ satisfying every clause. Therefore $$h_{a^\star}(C_j)=1$$ for every $$j$$, and hence $$D_\varphi$$ is realizable by $$H_n$$.

Run the assumed proper realizable learner with $$\epsilon<1/m$$ and $$\delta=1/3$$. If $$\varphi$$ is satisfiable, then with probability at least $$2/3$$ it returns a proper hypothesis $$h\in H_n$$ whose error under $$D_\varphi$$ is less than $$1/m$$. But under $$D_\varphi$$, even one unsatisfied clause gives error at least $$1/m$$. Therefore the returned hypothesis must satisfy $$h(C_j) = 1$$ for every $$j$$.

Since the learner is proper, $$h=h_a$$ for some assignment $$a$$. Thus $$a$$ satisfies every clause of $$\varphi$$.

So we can decide 3-SAT as follows. Given $$\varphi$$, simulate samples from $$D_\varphi$$, run the proper learner, and check whether the returned proper hypothesis labels every clause by $$1$$. If $$\varphi$$ is satisfiable, this accepts with probability at least $$2/3$$. If $$\varphi$$ is unsatisfiable, no assignment satisfies every clause, so no proper hypothesis can label every $$C_j$$ by $$1$$, and the algorithm never accepts. Thus 3-SAT would be in RP.

On the other hand, if we allow improper learning, then this becomes efficiently learnable in the agnostic case. Let $$F_n=\{0,1\}^{X_n}$$ be the class of all Boolean functions on $$X_n$$ and run proper learning wrt this class. Since the original class is a subset, this would give us an improper learner. Learning wrt $$F_n$$ is easy because first of all $$\log \lvert F_n\rvert = \lvert X_n\rvert = O(n^3)$$

Thus ERM suffices for learning. Now given a labeled sample of clauses, running ERM over $$F_n$$ is trivial. For each clause $$C\in X_n$$, look at all sample points equal to $$C$$ and label $$C$$ by the empirical majority label, breaking ties arbitrarily. This gives a lookup table $$\hat{g}:X_n\rightarrow \{0,1\}$$.

Therefore, assuming $$\mathrm{NP}\nsubseteq \mathrm{RP}$$, efficient improper agnostic learning does not imply efficient proper realizable learning.


[^dnf-to-cnf]: This works because in Boolean algebra, AND distributes over OR's.
