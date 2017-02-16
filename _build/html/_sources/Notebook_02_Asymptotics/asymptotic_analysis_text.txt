
Asymptotic analysis and consistency
===================================

**Date: November 2016**

In this notebook we discuss the consistency of decision rules. After a
general introduction we restrict attention to empirical loss minimizing
decision rules, establish sufficient conditions for consistency to hold
and discuss the rate at which the convergence takes place.

-  Consistency of a decision rule states that the corresponding
   empirical loss converges to the true loss in probability as the
   sample size grows to infinity.

-  In the case of empirical loss minimization, consistency is tightly
   linked to the uniform law of large numbers that we present in detail.

-  We introduce tail and concentration bounds to establish the uniform
   law of large numbers and analyze the (non-asymptotic) rate at which
   the convergence takes place.

Introduction
------------

Knowing the true mechanism :math:`P`, while one faces a decision problem
:math:`(\mathcal{H},L,\mathcal{A})`, naturally leads to the best in
class action :math:`a^{*}_{L, P, \mathcal{A}}` as the optimal action
within the prespecified class :math:`\mathcal{A}`. Although in reality
we do not know :math:`P`, it is illuminating to consider this case first
as we naturally require that at least with a sufficiently large (in
fact, infinite) sample, the true mechanism---from a statistical point of
view---can be revealed. The property that guarantees this outcome is the
ergodicity of the assumed statistical models.

This argument suggests to use the best in class action,
:math:`a^{*}_{L, P, \mathcal{A}}`, as the optimal large sample solution.
In other words, from any sensible decision rule a minimal property to
require is that as the sample size grows, the distribution of the
decision rule converges to a degenerate distribution concentrated at the
point :math:`a^{*}_{L, P, \mathcal{A}}`.

As an example, consider again the MLE estimator in the coin tossing
problem for which the above property is satisfied. The following figure
represents how the different confidence bands associated with the
distribution of the action evolve as the sample size goes to infinity.
Apparently, for sufficiently large sample sizes, the confidence bands
concentrate around the true value, which, due to the correct
specification is equivalent to the best in class action.

.. image:: ./asymptotic_cointoss_consistency.png
   :alt: 

This property of the decision rule is called **consistency**. One of the
main objectives of this notebook is to investigate the conditions under
which it can be established.

Given that asymptotically the true :math:`P` can be learned, a natural
question to ask is why not to set :math:`\mathcal{A}` large enough so as
to guarantee :math:`\gamma(P)\in\mathcal{A}`, i.e. a correctly specified
model. We will see the sense in which requiring consisteny ties our
hands in terms of the "size" of :math:`\mathcal{A}`.

Although it is hard to provide generally applicable sufficient
conditions for consistency, roughly speaking, we can identify two big
classes of decision rules for which powerful results are available.

-  Bayes decision rules
-  Frequentist decision rules building on the empirical distribution
   :math:`P_n`, where

.. math:: P_n(z) : = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\{Z_i \leq z\}

and :math:`\mathbf{1}\{ A \}` is the indicator function of the set
:math:`A`.

In this notebook we will focus exclusively on the latter approach
considering decision rules which assign actions based on minimizing the
empirical analog of the population loss. Hence, the procedure is
labelled by the name: *empirical loss minimization* or *analog
estimation*.

Consistency of decision rules
-----------------------------

To start out our asymptotic inquiry, we first define consistency in a
more precise manner than we did before. There are two, slightly
different notions depending on the tractability and objectives of the
decision problem at hand.

-  **Consistency in terms of the loss function:** a decision rule,
   :math:`d: \mathcal{S} \mapsto \mathcal{A}`, is consistent in terms of
   the loss function relative to :math:`(\mathcal{H}, \mathcal{A})`, if
   for all :math:`P \in \mathcal{H}`,

.. math::  L(P, d(Z^n)) \ \ \underset{n \to \infty}{\overset{\text{P}}{\rightarrow}} \ \ \inf_{a \in \mathcal{A}}L(P, a) = L\left(P, a^{*}_{L, P,\mathcal{A}}\right).

-  **Consistency in terms of the action:** a decision rule,
   :math:`d: \mathcal{S} \mapsto \mathcal{A}`, is consistent in terms of
   the action relative to :math:`(\mathcal{H}, \mathcal{A})`, if for all
   :math:`P \in \mathcal{H}`,

.. math::  P\left\{z^{\infty} \big| \lim_{n \to \infty} m\left(d(z_n), a^*_{L, P, \mathcal{A}}\right) > \epsilon\right\} = 0 \quad \text{for} \quad \forall\epsilon>0, 

where :math:`m(\cdot, \cdot)` is a metric on the action space
:math:`\mathcal{A}`. The necessary condition---in the case of analog
estimators---for this notion of consistency is the identifiability of
:math:`a^*_{L, P, \mathcal{A}}` that we define as follows:

**Identification:** :math:`a^*_{L, P, \mathcal{A}}` is identified
relative to :math:`(P, \mathcal{A})` if :math:`a^*_{L, P, \mathcal{A}}`
is the unique minimizer of :math:`L(P, \cdot)` over :math:`\mathcal{A}`.

Under identification, the two notions are equivalent. As the above
definitions suggest, however, the former is more general to the extent
that it also allows for partially identified statistical models. Unless
otherwise noted, we will work with consistency in terms of the loss
function and call it simply *consistency*.

In the above definitions, the set of assumed statistical models
:math:`\mathcal{H}` plays a key role: it outlines the set of
distributions under which the particular notion of convergence is
required. Ideally, we want convergence under the true distribution,
however, because we do not know :math:`P`, the "robust" approach is to
be as agnostic as possible regarding :math:`\mathcal{H}`. The central
insight of statstical decision theory is to highlight that this approach
is not limitless: one needs to find a balance between
:math:`\mathcal{H}` and :math:`\mathcal{A}` to obtain decision rules
with favorable properties.

Consistency has strong implications for the risk functional in large
samples: the degenerate limiting distribution of the decision rule
implies that asymptotically the variability of the decision rule
vanishes and so

.. math::  R_{\infty}(P,d) \overset{a.s.}{=} L\left(P, a^{*}_{L, P,\mathcal{A}}\right). 

Uniform law of large numbers
----------------------------

As mentioned before, in this notebook we are focusing on decision rules
arising from empirical loss minimization. The basic idea behind this
approach is to utilize the consistency definitions directly, but instead
of using the quantity :math:`L(P, a)` that we cannot actually evaluate,
subsitute the empirical distribution into the loss function and work
with the empricial loss :math:`L(P_n, a)` instead.

*Remark:* The empirical loss is not defined, if :math:`P_n` is not in
the domain of :math:`L`---as is the case, for example, with
non-parametric density estimation. Then, one has to either extend the
domain of :math:`L` or map :math:`P_n` to the original domain of the
loss function :math:`L`. An excellent discussion can be found in Manski
(1988).

Given that the loss function is continuous in its first argument, the
law of large numbers (LLN), accompanied with the continuous mapping
theorem, implies

.. math::  L(P_n, a) \quad  \underset{n \to \infty}{\to} \quad L(P, a) 

that is, for a *fixed action* :math:`a`, the emprical loss converges to
the true loss as the sample size goes to infinity. Notice, however, that
consistency

-  is not a property of a given action, but a whole decision rule, more
   precisely, it is about the convergence of the sequence
   :math:`\{d(z^n)\}_{n\geq 1} \subseteq \mathcal{A}`
-  requires convergence to the *minimum* loss (within
   :math:`\mathcal{A}`) that we ought to take into account while
   generating decision rules

In the spirit of the analogy principle, decision rules minimizing the
*empirical loss* can be written as follows

.. math::  d(z^n) := \arg\min_{a\in\mathcal{A}} L(P_n, a), 

where the dependence on the sample is embodied by the empirical
distribution :math:`P_n`. The heuristic idea is that because
:math:`d(z^n)` minimizes :math:`L(P_n, \cdot)`, and :math:`P_n`
converges to :math:`P`, then "ideally" :math:`d(z^n)` should go to
:math:`a^{*}_{L, P, \mathcal{A}}`, the minimizer of :math:`L(P, a)`, as
the sample size grows.

What do we need to ensure this argument to hold? First, notice that the
standard law of large numbers is inadequate for this purpose. In order
to illustrate this, we turn now to a concept closely related to
empirical loss minimization: generalization.

Generalization
~~~~~~~~~~~~~~

Using the empirical loss as a substitute for the true loss while
determining the decision rule, a critical question is how much error do
we introduce with this approximation. The object of interest regarding
this question is the *excess loss* for a given action :math:`a` and
realization of the sample :math:`z^n`

.. math::  \left| L(P_n, a) - L(P, a) \right|. 

We say that the action :math:`a` **generalizes** well from a given
sample :math:`z^n`, if the corresponding excess loss is small. Taking
the absolute value is important: :math:`L(P_n, a)` can easily be smaller
than :math:`L(P, a)` as it is in the case of overfitting.

Due to :math:`P_n`'s dependence on the particular sample, however, the
excess loss (for a given :math:`a`) is a random variable. It might be
small for a given sample :math:`z^n`, but what we really need in order
to justify the method of empricial loss minimization is that the excess
loss is small for "most samples", or more precisely

.. math::  P\left(z^n : \left| L(P_n, a) - L(P, a) \right| > \delta \right) \quad \text{is small.} 

Since :math:`L` is continuous in its first argument and
:math:`P_n \to P`, this so called **tail probability** converges to zero
for any fixed :math:`a\in\mathcal{A}`. The figure below displays such
tail probabilities for the MLE estimator in the coin tossing example
(with quadratic loss) for different actions as functions of the sample
size.

One can see that even though the tail probabilities converge to
:math:`0` for all :math:`a\in\mathcal{A}`, the *rate of convergence*
depends on the particular :math:`a`. In other words,
:math:`\forall \varepsilon>0` and :math:`\forall a\in\mathcal{A}`, there
is an *action dependent* minimum sample size
:math:`N(a, \varepsilon, \delta)`, that we need in order to guarantee
that the corresponding tail probability falls below :math:`\varepsilon`.

Suppose now that we have a decision rule and look at its induced
sequence of actions :math:`\{a_n\}_{n\geq 1}` for a fixed realization
:math:`z^{\infty}`. Again, for consistency, we need the associated
sequence of tail probabilities converging to zero for "most
:math:`z^{\infty}`".

-  When the set :math:`\mathcal{A}` has finitely many elements, this is
   not a problem: we can simply define

.. math:: N(\varepsilon, \delta) : = \max_{a\in\mathcal{A}} N(a, \varepsilon, \delta)

and observe that for all :math:`n\geq N(\varepsilon, \delta)`, the tail
probabilities are smaller than :math:`\varepsilon`, for all
:math:`a\in \{a_n\}_{n\geq 1} \subseteq \mathcal{A}`.

-  However, because the critical sample size
   :math:`N(a, \varepsilon, \delta)` depends on the action, if
   :math:`\mathcal{A}` is "too big", it is possible that there is no
   :math:`n` so that :math:`n > N(a_n, \varepsilon, \delta)`, hence
   :math:`L(P_n, a_n)` may never approach :math:`L(P, a_n)`.

A possible approach to avoid this complication is to require the
existence of an integer :math:`N(\varepsilon, \delta)` *independent of*
:math:`a`, such that for all sample sizes larger than
:math:`N(\epsilon, \delta)`

.. math::  \lim_{n\to \infty} P\left\{\sup_{a \in\mathcal{A}} \left| L(P_n, a) - L(P, a) \right| > \delta\right\} = 0. 

This notion is called the **uniform law of large numbers** referring to
the fact that the convergence is guaranteed simultaneously for all
actions in :math:`\mathcal{A}`.

.. figure:: ./asymptotic_cointoss_tail.png
   :alt: 

A special case -- plug-in estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are special cases where we do not need to worry about uniformity.
When the true loss function has a minimizer
:math:`a^{*}_{L, P, \mathcal{A}}` that admits a closed form in the sense
that it can be expressed as a continuous function of :math:`P`, the
empirical loss minimizer is simply a sample analog of
:math:`a^{*}_{L, P, \mathcal{A}}`. Effectively, there is no need to
minimize the emprical loss, so in spirit, it is as if we kept a
particular action fixed. In these cases the standard LLN is sufficient
to establish consistency.

-  *Sample mean:* Consider the quadratic loss function with the feature
   :math:`\gamma(P) = E[Z]`, i.e. the loss is
   :math:`L(P,a) = E[(a - E[Z])^2]`. Evidently, the minimizer of this
   quantity is :math:`E[Z]`. Although we could proceed by minimizing the
   empricial loss to derive the decision rule, we do not need to do
   that, because we know up front that it is equal to the sample analog
   of :math:`E[Z]`. The decision rule is the plug-in estimator
   :math:`d(z^n) = \frac{1}{n}\sum_{i=1}^{n}z_i`.

-  *OLS estimator:* Suppose that :math:`Z=(Y, X)` and consider the
   quadratic loss :math:`L(P, a) = E[(Y-a)^2 | X]`, where
   :math:`E[\cdot | X]` is the expectation operator conditioned on
   :math:`X`. Assume that :math:`(Y, X)` follows a multivariate normal
   distribution, then the minimizer of :math:`L` is given by
   :math:`a = E\left[(X'X)^{-1}\right]E[X'Y]`. Consequently, there is no
   need for explicit minimization, we can use the least squares
   estimator as a plug-in estimator for :math:`a`.

More generally, as the empirical loss minimizer is itself a functional,
if the "argmin" functional can be shown to be continuous on the space of
empirical loss functions, then an application of the continuous mapping
theorem together with the Glivenko-Cantelli theorem (i.e.
:math:`\lVert P_n - P \rVert_{\infty} \to 0`) would yield consistency.
For reference, see van der Vaart (2000).

Uniform law of large numbers and consistency
--------------------------------------------

In more general nonlinear models, however,
:math:`a^{*}_{L, P, \mathcal{A}}` has no analytical form, so we cannot
avoid minimizing the empirical loss in order to derive the decision
rule. In these cases, guaranteeing the validity of uniform convergence
becomes essential. In fact, it can be shown that consistency of any
decision rule that we derive by minimizing some empirical loss function,
is *equivalent with* the uniform LLN.

**Vapnik and Chervonenkis (1971)**: Uniform convergence

.. math::  \lim_{n\to \infty} P\left\{\sup_{a \in\mathcal{A}} \left| L(P_n, a) - L(P, a) \right| < \varepsilon\right\}= 1 

for all :math:`\varepsilon >0` is *necessary and sufficient* for
consistency of decision rules arising from empirical loss minimization
over :math:`\mathcal{A}`.

Although this characterization of consistency is theoretically
intriguing, it is not all that useful in practice unless we find
easy-to-check conditions which ensure uniform convergence of empirical
loss over a particular function class :math:`\mathcal{A}`. As a
preparation to discuss these conditions, introduce first some notation

Almost all loss functions used in practice can be cast in the following
form. There exist

-  :math:`l : \mathcal{A}\times Z \mapsto R^{m}` and
-  a *continuous* :math:`r : R^{m} \mapsto R_+` such that

.. math::  L(P, a) = r\left( \int_Z l(a, z)dP (z)\right) 

Continuity of :math:`r` implies that in order to establish consistency
it is enough to investigate the properties of the limit

.. math::  P\left( \sup_{a\in\mathcal{A}} \left| \int_Z l(a, z)dP_n (z) - \int_Z l(a, z)dP(z)\right| > \varepsilon \right) \quad  \underset{n \to \infty}{\to} \quad  0 \quad\quad \forall \varepsilon > 0

Formally, this requires that the class of functions
:math:`\mathcal{L}_{\mathcal{A}}:= \left\{l(a, \cdot) : a \in \mathcal{A} \right\}`
is a **Glivenko-Cantelli class** for :math:`P`.

--------------

**Definition (Glivenko-Cantelli class):** Let :math:`\mathcal{G}` be a
class of integrable real-valued functions of the random variable
:math:`Z` having distribution :math:`P`. Consider the random variable

.. math::  \Vert P_n - P \Vert_{\mathcal{G}} : = \sup_{f\in\mathcal{G}} \left| P_n g - Pg \right| 

We say that :math:`\mathcal{G}` is a Glivenko-Cantelli class for
:math:`P`, if :math:`\Vert P_n - P \Vert_{\mathcal{G}}` converges to
zero in probability as :math:`n\to\infty`.

--------------

A useful decomposition highlights the importance of the uniform law of
large numbers in the case of empirical loss minimization. We would like
to know the difference between the true loss of our estimator and the
true loss of the best in class action. Denote the analog estimate as
:math:`\hat{a} = d(z^n)` and the best in class action as :math:`a^*`.

.. math::  Pl_{\hat{a}} - Pl_{a^*} = (Pl_{\hat{a}} - P_n l_{\hat{a}}) + (P_n l_{\hat{a}} - P_n l_{a^*})  + (P_n l_{a^*} - P l_{a^*}) 

The last term on the right hand side is governed by the Law of Large
Numbers, the middle term is necessarily weakly positive by definition of
the empirical loss minimizing decision rule, and the first term is
governed by the Uniform Law of Large Numbers. Hence, if we want to
control the excess risk -- the left hand side -- then we necessarily
have to control the right hand side.

Sufficient conditions for ULLN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variables of the form :math:`\Vert P_n - P \Vert_{\mathcal{G}}` are
ubiquitos in statistics and econometrics and there are well-known
sufficient conditions that guarantee its convergence.

For parametric estimation, probably the most widely used (at least in
econometrics) sufficient condition includes the following three
assumptions (or some form thereof). Suppose that the action space
:math:`\mathcal{A}` is indexed by a finite dimensional vector
:math:`\theta\in\Theta \subset \mathbb{R}^{p}`. If

-  :math:`\Theta` is *compact*
-  :math:`l(\theta, z)` is a function such that :math:`l(\cdot, z)` is
   *continuous* on :math:`\Theta` with probability one
-  :math:`l(\theta, z)` is *dominated* by a function :math:`B(z)`, i.e.
   :math:`|l(\theta, z)|\leq B(z)` for all :math:`\theta\in\Theta`, such
   that :math:`\mathbb{E}[B(Z)]<\infty`

then
:math:`\mathcal{L}_{\mathcal{A}} := \left\{l(a, \cdot) : a \in \mathcal{A} \right\}`
is a Glivenko-Cantelli class for :math:`P` and so the estimator that it
represents is consistent. Among others, these assumptions are the bases
for consistency of the (Quasi-)Maximum Likelihood (White, 1994) and the
Generalized Method of Moments estimators (Hansen, 1982).

A somewhat different approach focuses on the "effective size" of the
class :math:`\mathcal{L}_{\mathcal{A}}` and frames sufficient conditions
in terms of particular complexity measures such as the Rademacher
complexity, Vapnik-Chervonenkis dimension, covering/packing numbers,
etc. The idea is to find tail bounds for the probability that
:math:`\Vert P_n - P \Vert_{\mathcal{L}_{\mathcal{A}}}` deviates
substantially above the complexity of :math:`\mathcal{L}_{\mathcal{A}}`.
In the following section we present the main ideas behind the derivation
of non-asymptotic tail bounds.

**An interesting necessary condition:** If
:math:`\mathcal{L}_{\mathcal{A}}` is a collection of indicator functions
and the data generating process is assumed to be i.i.d., a necessary and
sufficient condition for distribution free uniform convergence is that
the Vapnik-Chervonenkis complexity of :math:`\mathcal{L}_{\mathcal{A}}`
is finite.

Non-asymptotic bounds
---------------------

We saw that in order to establish consistency for analog estimators we
had to establish uniform convergence of the empirical loss to the true
loss over the entire action space. In this section, we present an
approach to the laws of large numbers (uniform or "standard"), which
builds on a finite sample perspective.

For a given finite sample size, we study the concentration of the
centered averages around their means. Ensuring that the centered average
goes to a degenerate distribution on its mean---"perfectly" concentrates
around it---will establish the law of large numbers.

Studying concentration and tail bounds for each finite sample size has
the advantage that we will get rates of convergence as a byproduct. The
rate of convergence will give us information regarding the minimum
sample size above which the asymptotic results are going to be good
approximations.

Tail and Concentration Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Throughout the section we should keep in mind the characterization of
consistency in the case of analog estimators. This was equivalent to
uniform convergence which we repeat here

.. math::  P\left( \sup_{a\in\mathcal{A}} \left| \int_Z l(a, z)dP_n (z) - \int_Z l(a, z)dP(z)\right| > \varepsilon \right) \quad  \underset{n \to \infty}{\to} \quad  0 \quad\quad \forall \varepsilon > 0. 

For each action the loss is a just a (measurable) function of the random
variable :math:`Z`, :math:`l_a(Z):=l(a, Z)`. Hence, we need to study the
behavior of the empirical process indexed by the function class
:math:`\mathcal{L}_{\mathcal{A}}`

Note, that :math:`P_n l_a` is a random variable because the empirical
distribution is a function of the random sample. :math:`Pl_a` is just
the expectation of :math:`l_a(Z)` and hence it is a non-random scalar.

First, we study how the concentration measures work for a single random
variable. This is going to parallel the law of large numbers. Second, we
study concentration measures taking place uniformly over a class of
random variables. This is going to parallel the uniform law of large
numbers. For ease of notation we present the results for a general class
of measurable functions :math:`g \in \mathcal{G}` -- but remember that
in almost all applications we will substitute it for
:math:`\mathcal{L}_{\mathcal{A}}`.

Markov's inequality
~~~~~~~~~~~~~~~~~~~

Markov's inequality states that for any non-negative scalar random
variable :math:`Z` and :math:`t>0` we have

.. math::  P\{Z \geq t\}\leq \frac{\mathbb{E}[Z]}{t}. 

The idea behind this inequality is fairly simple and in fact it follows
from the following obvious inequality

.. math::  \mathbf{1}\{Z\geq t\}\leq \frac{Z}{t} 

illustrated by the figure below. Clearly, for any probability measure
:math:`P` over :math:`Z` with a finite first moment Markov's inequality
can be established.

It follows from Markov's inequality that for any strictly monotonically
increasing non-negative function :math:`\phi` and any random variable
:math:`Z` (not necessarily non-negative) we have that

.. math::  P\{Z \geq t\} = P\{\phi(Z) \geq \phi(t)\} \leq \frac{\mathbb{E}[\phi(Z)]}{\phi(t)}. 

Taking the centered random variable :math:`| Z - \mathbb{E}[Z]|` and
:math:`\phi(x) = x^q` for some :math:`q>0` leads to tail bounds
expressed in terms of the moments of the random variable :math:`Z`.

.. math::  P\{| Z - \mathbb{E}[Z]| \geq t\} \leq \frac{\mathbb{E}[| Z - \mathbb{E}[Z]|^q]}{t^q}. 

Note that for :math:`q = 2`, this form delivers the Chebyshev inequality
(see the right panel of the Figure). This approach to bounding tail
probabilities is quite general. Controlling higher order moments leads
to (weakly) tighter bounds on the tail probabilities.

.. figure:: ./asymptotic_markov_chebyshev.png
   :alt: 

Chernoff bounds
~~~~~~~~~~~~~~~

A related idea is at the core of the so called Chernoff bound. For that
one takes the transformation :math:`\phi(x) = e^{\lambda x}` applied to
the centered random variable :math:`(Z - \mathbb{E}[Z])` which yields

.. math::  P\{(Z - \mathbb{E}[Z]) \geq t\} \leq \frac{\mathbb{E}\left[e^{\lambda(Z - \mathbb{E}[Z])}\right]}{e^{\lambda t}}. 

Minimizing the bound over :math:`\lambda` (provided the moments exist)
would lead us to the Chernoff bound

.. math::  \log P\{(Z - \mathbb{E}[Z]) \geq t\} \leq - \sup_{\lambda} \left\{\lambda t - \log \mathbb{E}\left[e^{\lambda(Z - \mathbb{E}[Z])}\right]\right\} 

If the centered random variable is an iid sum of other random
variables---as in the case of :math:`(P_n g - P g)`---the Chernoff bound
gains additional structure.

Hoeffding bounds
~~~~~~~~~~~~~~~~

Suppose first that the sample :math:`z^n` is generated by an iid process
and recall that
:math:`(P_n g - P g) = \frac{1}{n}\sum_{i=1}^n g(Z_i) - \mathbb{E}[g(Z)]`.
Bounding the tails via Chernoff's method yields

.. math::  P\left\{\left(\frac{1}{n}\sum_{i=1}^n g(Z_i) - \mathbb{E}[g(Z)]\right) \geq t\right\} \leq \frac{\mathbb{E}\left[e^{\lambda \frac{1}{n}\sum_{i=1}^n \left(g(Z_i) - \mathbb{E}[g(Z)]\right)}\right]}{e^{\lambda t}} = e^{-\lambda t} \prod_{i=1}^n \mathbb{E}\left[e^{\lambda \frac{1}{n}\left(g(Z_i) - \mathbb{E}[g(Z)] \right)} \right] 

where the equality follows from independece. Hence, the problem of
deriving a tight bound boils down to bounding the moment generating
function of

.. math::  \frac{1}{n}\left(g(Z_i) - \mathbb{E}[g(Z)] \right). 

For our purposes a particularly important class of random variables are
the so called sub-Gaussian variables.

**Definition.** A random variable :math:`Z` is called sub-Gaussian if
there is a positive number :math:`\sigma` such that

.. math::  \mathbb{E}\left[e^{\lambda \left(Z - \mathbb{E}[Z] \right)} \right] \leq e^{\sigma^2\lambda^2/2}

for all :math:`\lambda \in \mathbb{R}`.

Remark: A Guassian variable with variance :math:`\sigma^2` is
sub-Gaussian with parameter :math:`\sigma`. There are other non-Guassian
random variables which are sub-Gaussian -- for example all bounded
random variables, Rademacher variables

**Theorem.** (Hoeffding) Let the variables :math:`g(Z_i)` be iid and
sub-Gaussian with parameter :math:`\sigma`. Then, for all :math:`t>0` we
have that

.. math:: P\left\{\left|\frac{1}{n}\sum_{i=1}^n g(Z_i) - \mathbb{E}[g(Z)]\right| \geq t\right\} \leq 2\exp\left\{-\frac{t^2 n}{2\sigma^2}\right\}.

For a single random variable we see how the Hoeffding inequality is at
the core of the law of large numbers. In fact it gives an exponential
rate of convergence.

However, in order to talk about concentration properties of decision
rules, :math:`d: \mathcal{S}\mapsto\mathcal{A}` we would like to make
statements about tail probabilities uniformly over a class of
actions---and hence uniformly across :math:`\mathcal{L}_{\mathcal{A}}`.
This leads us to uniform bounds.

Uniform Tail and Concentration Bounds for classes of finite cardinality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first simple example we consider uniform tail bounds over sets of
finite cardinality. Let
:math:`\mathcal{G} = \{g_j : j = 1, \ldots, J\}`. A conservative
estimate of the uniform tail bound in this case is the union bound.

**Corollary.** (Hoeffding union bound) Let the variables
:math:`\{g_j(Z_i): j=1,\ldots, J\}` be iid and sub-Gaussian with common
parameter :math:`\sigma`. Then, for all :math:`t > 0` we have that

.. math::  P\left\{\sup_{g\in\mathcal{G}}\left| \frac{1}{n}\sum_{i=1}^n g(Z_i) - \mathbb{E}[g(Z)]\right| \geq t \right\} \leq \sum_{j = 1}^{J} P\left\{\left|\frac{1}{n}\sum_{i=1}^n g_j(Z_i) - \mathbb{E}[g_j(Z)]\right| \geq t \right\} \leq J 2\exp\left\{-\frac{t^2 n}{2\sigma^2}\right\}. 

The difference between the uniform and individual Hoeffding bounds is
just a scaling factor, :math:`J`. If there is only finitely many actions
in :math:`\mathcal{A}`, then of course the function class
:math:`\mathcal{L}_{\mathcal{A}}` has finite cardinality. Our objective,
however, is to extend the above analysis to action spaces with infinite
cardinality. To this end, we need to find a better "measure" of the size
of a function space than its cardinality. Next, we introduce one such
complexity measure which proves to be extremely useful in the case of
characterizing tail bounds for analog estimators.

Uniform tail bounds for classes of infinite cardinality - Rademacher complexity
-------------------------------------------------------------------------------

In order to work with sets of infinitely many functions we would like to
capture the size of these infinite sets for the purpose of statistical
analysis. One such measure of statistical size is the Rademacher
complexity of a class of real-valued functions.

For a given realization of a sample, :math:`z^n`, of size :math:`n`
consider the the set of vectors

.. math:: \mathcal{G}(z^n) := \left\{ (g(z_1), \ldots, g(z_n)) \in \mathbb{R}^n \mid g \in \mathcal{G} \right\}.

This is the number of ways one can label points of a sample using
functions in the class :math:`\mathcal{F}`. It is usually called the
projection of the function class :math:`\mathcal{G}` to the sample
:math:`z^n`. The **empirical Rademacher complexity of**
:math:`\mathcal{G}` for fixed :math:`z^n` is defined as

.. math:: R\left(\mathcal{G}(z^n) \right) := \mathbb{E}_{\epsilon}\left[\sup_{g\in \mathcal{G}}\Big| \frac{1}{n}\sum_{i=1}^n \epsilon_i g(z_i) \Big| \right],

where the expectation is taken with respect to the iid Rademacher random
variables, :math:`\epsilon_i` which take value in :math:`\{-1, 1\}` with
equal probability.

The **Rademacher complexity of** :math:`\mathcal{L}` at sample size
:math:`n` is defined as

.. math:: R_n\left(\mathcal{G}\right) := \mathbb{E}_{Z^n} \Big[R\left(\mathcal{G}(z^n) \right)\Big].

The Rademacher complexity has an intuitive interpretation. It is the
average of the maximum correlations between the vectors
:math:`\big(g(z_1), \ldots, g(z_n)\big)` and the pure noise vector
:math:`\big(\epsilon(z_1), \ldots, \epsilon(z_n)\big)`. The function
class :math:`\mathcal{G}` is too "large" for statistical purposes, if we
can always choose a function, :math:`g\in\mathcal{G}` that has high
correlation with a randomly drawn noise vector (Wainwright, 2015).

Illustration of Rademacher complexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following we consider two classes of functions. Although both
classes can be parametrized by a *single* free parameter, we will see
that one of them has drastically smaller Rademacher complexity than the
other.

Coin tossing example revisited
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider again the coin tossing example with the quadratic loss function
discussed before. In this case :math:`Z^n` is an iid sample from a
Binomial distribution parametrized with :math:`p\in[0, 1]`. The class of
functions is given by the quadratic class

.. math::  \mathcal{L}_{\mathcal{A}} := \{l_a : l_a(z) = (z - a)^2, \ \  a \in [0, 1]\subseteq \mathbb{R}\} 

Below we see how the Rademacher compexity of
:math:`\mathcal{L}_{\mathcal{A}}` converges to zero as the sample size
grows to infinity.

.. image:: ./asymptotic_rademacher_cointoss.png
   :width: 640
   :height: 410
   :align: center

Sinusoid classification
^^^^^^^^^^^^^^^^^^^^^^^

Next, we consider a set of classifier functions where the classification
boundary is given by a sine function.

.. math::  \mathcal{L}_{\mathcal{A}} := \Big\{ \mathbb{1}\{sin(az) \geq 0\} - \mathbb{1}\{sin(az) < 0\} : a \in \mathbb{R}_+ \Big\} 

For better illustration, in this case we consider the empirical
Rademacher complexity for a fixed realization of the sample,
:math:`z^n`.

*Remark:* In order to have a closed form solution for the optimal
classifier we are selecting the sample at convenient points. This is
without loss of generality and useful for illustrative purposes.

.. image:: ./asymptotic_rademacher_sinusoid.png 
   :width: 640
   :height: 410
   :align: center


As we can see, by choosing a sufficiently high frequency we can always
find a curve which classifies the data perfectly. Consequently, the
Ramdemacher complexity always takes its maximum irrespective of the
sample size. For statistical purposes the family of sine curves is too
complex. This example highlights that for general non-linear functions
the number of free parameters---here only one---does not correspond to
the complexity of the functions class---which is infinity.

Uniform bounds using the Rademacher complexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the introduced concepts we are in the position to present an
important concentration inequality for classes of infinite cardinality.

**Theorem:** For uniformly bounded functions
:math:`\lvert g \rvert_{\infty} < B, \ \ \forall g \in\mathcal{G}` and
for each :math:`\delta>0` we have that

.. math::  P \Big\{ \Vert P_n - P \Vert_{\mathcal{G}} \geq  2R_n(\mathcal{G}) + \delta \Big\} \leq 2 \exp\Big\{- \frac{2 n \delta^2}{B^2} \Big\}. 

and for the empirical Rademacher complexity

.. math::  P\Big\{ \Vert P_n - P \Vert_{\mathcal{G}} \geq 2 R\left(\mathcal{G}(Z^n)\right) + \delta \Big\} \leq 2 \exp\Big\{-\frac{n \delta^2}{4 B^2} \Big\}. 

It is apparent from the above inequality that the tightness of the
finite sample uniform bounds will be (partly) determined by the
Rademacher complexity. The change in Rademacher complexity as the sample
size grows will determine the (non-asymptotic) rate of convergence.
Hence, if :math:`R_n(\mathcal{G}) = o(1)` then
:math:`\Vert P_n - P \Vert_{\mathcal{G}} \to 0` or put it differently,
:math:`\mathcal{G}` is Glivenko-Cantelli.

Strongly related to the tail bounds one can bound the expectation of the
supremum of an empirical process using Rademacher complexity. This is
often called as symmetrization inequality.

**Theorem:** For any class of :math:`P`-integrable functon class
:math:`\mathcal{G}` we have that

.. math:: \mathbb{E}_{Z^n}\Big[\Vert P_n - P \Vert_{\mathcal{G}} \Big] \leq 2R_n(\mathcal{G}) = 2 \mathbb{E}_{Z^n}\Big[R\left(\mathcal{G}(Z^n)\right)\Big]. 

Unfortunately, computing the Rademacher complexity directly is only
feasible in special cases. There are various techniques however, which
give bounds on the Rademacher complexity. For different classes of
functions different techniques prove useful, so one usually proceeds on
a case-by-case basis. The most common ways of bounding the Rademacher
complexity are via the *Vapnik-Chervonenkis dimension* for binary
functions and via *metric entropy* for bounded real valued functions.

--------------

References
~~~~~~~~~~

Bousquet, O., Boucheron, S., & Lugosi, G. (2004). Introduction to
statistical learning theory. In Advanced lectures on machine learning
(pp. 169-207). Springer Berlin Heidelberg.

Chervonenkis, A. and Vapnik, V. (1971). Theory of uniform convergence of
frequencies of events to their probabilities and problems of search for
an optimal solution from empirical data. Automation and Remote Control,
32, 207-217.

Hansen, L. P. (1982). Large sample properties of generalized method of
moments estimators. Econometrica: Journal of the Econometric Society,
1029-1054.

Manski, C. F. (1988). Analog estimation methods in econometrics. Chapman
and Hall.

Van der Vaart, A. W. (2000). Asymptotic statistics (Vol. 3). Cambridge
University Press.

White, Halbert (1994), Estimation, Inference and Specification Analysis
(Econometric Society Monographs). Cambridge University Press.

---------

The code for the simulations and generating the graphs can be found  `here.
<https://github.com/QuantEcon/econometrics/blob/master/_build/html/Notebook_02_Asymptotics/asymptotic_analysis_code.html>`_.
