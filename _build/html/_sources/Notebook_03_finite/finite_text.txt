
Coping with finite samples
==========================

**February 2017**

From asymptotics to finite samples
----------------------------------

In notebook {reference asymptotics} we took an asymptotic approach by
focusing on the behavior of the decision rule under the assumption that
the sample size is "almost infinity". We identified consistency and
fast rate of convergence as desirable features that any reasonable
decision rule should satisfy.

We drew the crucial conclustion that both consistency and the rate of
convergence are driven by the *complexity* of the composite function
class :math:`\mathcal{L}_{\mathcal A}`. In particular, we have seen that
finiteness of model complexity is enough to ensure consistency and given
consistency the magnitude of complexity is inversly related to the
decision rule's rate of convergence: the lower the complexity, the
faster the rate of convergence.

This notebook explores the implications of taking the finiteness of the
sample size seriously. We investigate the additional issues arising
relative to the asymptotic case and outline some approaches meant to
tackle these.

Decompositions of the risk functional
-------------------------------------

A consistent decision rule :math:`d` satisfies
:math:`L(P, d(z^n)) \overset{P}{\to} L(P, a^*_{L, P, \mathcal{A}})`,
so--by definition--the resulting loss functional has a degenerate
asymptotic distribution centered around the loss of the best-in-class
action. One might call this value "asymptotic risk". Being dependent on
the action space :math:`\mathcal A`, however, this asymptotic risk is
not necessarily zero. The global minimum of the loss function relates to
the true feature of the DGP that we denoted by
:math:`\gamma(P)=a^*_{L, P, \mathcal{F}}`, with :math:`\mathcal{F}`
being the set of all admissible actions.

The central objects of finite sample analysis concern how the finite
risk deviates from these two "asymptotic" values:

-  The deviation from the best-in-class loss is called **estimation
   error**:

.. math:: \mathcal E_d(P, \mathcal A, n) := R_n(P, d) - L\left(P, a^{*}_{L, P, \mathcal{A}}\right) 

-  The deviation from the global minimum is the so called **excess
   risk**:

.. math:: \mathcal{ER}_d(P, n) :=  R_n(P, d) - L\left(P, a^*_{L, P, \mathcal{F}} \right) 

Estimation-misspecification decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Naturally, the two objects are closely connected. In fact, decomposing
the excess risk using the estimation error highlights one of the most
important tensions underlying finite sample inference problems.

.. math::  R_n(P, d) - L\left(P, a^*_{L, P, \mathcal{F}}  \right) =  \underbrace{R_n(P, d) - L\left(P, a^{*}_{L, P, \mathcal{A}}\right)}_{\substack{\text{estimation error} \\ \text{random}}} + \underbrace{L\left(P, a^{*}_{L, P, \mathcal{A}}\right)- L\left(P, a^*_{L, P, \mathcal{F}}  \right)}_{\substack{\text{misspecification error} \\ \text{deterministic}}}. 

As we noted earlier the **misspecification error** is stemming from the
fact that the true feature might lie outside of the action space.
Intuitively, as we enlarge the action space :math:`\mathcal{A}`, the
misspecification error gets weakly smaller, while the estimation error
gets weakly larger. An ideal decision rule balances this trade-off.

Bias-volatility-misspecification decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To clarify why the estimation error might increase in the size of the
action space, we study a slightly finer decomposition of excess risk
bringing the "standard" bias-variance trade-off to bear. The starting
point of this decomposition is the observation that by assigning members
of :math:`\mathcal A` to *random* samples :math:`z^n` the decision rule
:math:`d` effectively induces a random variable on the action space.
Having said that, for each sample size :math:`n`, we can define the
*mean action* :math:`\bar d_n` by taking the expected value of
:math:`d(z^n)\in\mathcal A` across the different ensemble samples:

.. math:: \bar d_n = \int_{Z^n} d(z^n) \mathrm{d}P(z^n).

Notice that the mean action need not be in the decision problem's action
space :math:`\mathcal A`. Nonetheless, we can evaluate the loss at
:math:`\bar d_n` and use this value to decompose the excess risk's
estimation error component

.. math::  R_n(P, d) - L\left(P, a^*_{L, P, \mathcal{F}} \right) = \underbrace{R_n\left(P, d\right) - L\left(P, \bar d_n \right)}_{\text{volatility}} + \underbrace{L\left(P, \bar{d_n}\right) - L\left(P, a^{*}_{L,P,  \mathcal{A}}\right)}_{\text{bias}} + \underbrace{L\left(P, a^{*}_{L, P, \mathcal{A}}\right)- L\left(P,a^*_{L, P, \mathcal{F}}  \right)}_{\text{misspecification}} 

The only term that depends on the particular sample is the first one,
hence the name volatility. We prefer to view this term as a measure of
"instability" of the decision rule. It captures how sensitive is the
chosen action to small variations in the realized sample. Given this
interpretation, it is relatively straightforward to see the volatility
term's close connection with Rademacher complexity. Remember, the
Rademacher complexity of a function class essentially captures the
maximal pairwise correlation between elements of the class and
independent random noise. Large Rademacher complexity implies that the
function class can fit any random noise and hence we expect the chosen
element to be highly volatile.

Since the bias and misspecification terms are deterministic, it might be
tempting to pull them together and define a "total bias" component as is
typically done in the statistical learning literature [Abu-Mostafa-2012]_. Notice, however, that there is a key
difference between these two terms. While the bias term depends on the
sample size :math:`n`, the misspecification term remains constant given
that the action space :math:`\mathcal{A}` is kept fixed. In fact, we
know from our analysis in lecture {asymptotic lecture} that

-  If the decision rule :math:`d` is consistent relative to
   :math:`(\mathcal{H}, \mathcal{A})`, the *bias* converges to zero as
   :math:`n` goes to infinty.
-  If the decision rule :math:`d` is consistent relative to
   :math:`(\mathcal{H}, \mathcal{A})`, the *volatility* converges to
   zero as :math:`n` goes to infinty.

Later on in this notebook, we will see that nothing prevents the
statistician to flexibly adjust the range of a decision rule -- that is
the action space, :math:`\mathcal{A}`, where it can take values -- with
the quality and abundance of data. This way one can mitigate the
drawbacks of high volatility in small samples while also reduce the
misspecification error as the sample size grows and volatility becomes
less of an issue.

Illustration of the bias-volatility-misspecification decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**1. Quadratic loss**

The elements of the problem are

-  *Observable:* :math:`Z = (Y,X)`
-  *Loss function:* :math:`L(P, a) = \int_Z (y - a(x))^2 \mathrm{d}P(z)`
-  *Admissible space:*
   :math:`\mathcal{F}\equiv \{ a: x \mapsto y \ | \ a \ \text{is measurable}\}`
-  *True feature:*
   :math:`\gamma(P) = \mathbb{E}_P[Y|X] = \inf_{a\in \mathcal{F}} \ L(P, a)`

Correspondingly, the minimal loss can be written as

.. math:: L(P, a^*_{P, \mathcal{F}}) = \int_Z \underbrace{(y - \mathbb{E}_P[Y|X](x))^2}_{=\sigma^2_x \ \ \text{(noise)}} \mathrm{d}P(z) = \mathbb{E}_P[\sigma^2_x].

The loss evaluated at the best-in-class action,
:math:`a^*_{P, \mathcal{A}}(x)= \inf_{a\in\mathcal{A}} \ L(P, a)`, is

.. math::  L\left(P, a^*_{P, \mathcal{A}}\right) = \mathbb{E}_P[\sigma^2_x] + \int_Z \underbrace{\left[\mathbb{E}_P[Y|X](x) - a^*_{P, \mathcal{A}}(x)\right]^2}_{= \text{misspecification}^2_x} \mathrm{d}P(z) = L(P,a^*_{P, \mathcal{F}}) + \mathbb{E}_P\left[\text{misspecification}^2_x\right] 

and the loss evaluated at the average action
:math:`\bar d_n(x) := \int_{Z^n} a_{z^n}(x) \mathrm{d}P(z^n)` is

.. math:: L\left(P, \bar d_n\right) = L\left(P, a^*_{P, \mathcal{A}}\right)  + \int_Z \underbrace{\left[a^*_{P, \mathcal{A}}(x) - \bar d_n(x)\right]^2}_{= \text{bias}_x^2} \mathrm{d}P(z) = L\left(P, a^*_{P, \mathcal{A}}\right)  + \mathbb{E}_P\left[ \text{bias}_x^2 \right]

The volatility term is simply

.. math:: R_n(P, d) - L(P, \bar d_n) = \int_Z \left[\int_{Z^n} \left(a_{z^n}(x) - \bar d_n(x)\mathbf{1}(z^n)\right)^2\mathrm{d}P(z^n)\right]\mathrm{d}P(z)  = \mathbb{E}_P\left[\text{volatility}_x\right].

Therefore, the excess risk of a decision rule :math:`d` under the
quadratic loss is

.. math:: R_n(P, d) - L(P, a^*_{P, \mathcal{F}}) = \mathbb{E}_P\left[\text{misspecification}^2_x\right] + \mathbb{E}_P\left[\text{bias}_x^2\right] + \mathbb{E}_P\left[\text{volatility}_x\right].

**2. Relative entropy loss**

The elements of the problem are

-  *Observable:* :math:`Z \sim P`, where :math:`P` has the density
   :math:`p`
-  *Loss function:*
   :math:`L(P, a) = \int_Z p(z)\log \frac{p}{a}(z) \mathrm{d}z`
-  *Admissible space:* distributions on :math:`Z` for which density
   exists. Denote these densities by :math:`a(z)`
-  *True feature:* :math:`\gamma(P) = p(z)`

Then the minimal loss is zero and it is reached by :math:`a(z)=p(z)`,
i.e. :math:`L(P, a^*_{P, \mathcal{F}})=0`.

The loss evaluated at the best-in-class action
:math:`a^*_{P, \mathcal{A}}(z)= \inf_{a\in\mathcal{A}} \ L(P, a)`, is

.. math:: L\left(P, a^*_{P, \mathcal{A}}\right) = \int_Z \underbrace{\log\left(\frac{p}{a^*_{P, \mathcal{A}}}\right)(z)}_{= \text{misspecification}_z} \mathrm{d}P(z) = \mathbb{E}_P\left[\text{misspecification}_z\right]

and the loss evaluated at the average action
:math:`\bar d_n(z) := \int_{Z^n} a_{z^n}(z) \mathrm{d}P(z^n)` is

.. math::  L\left(P, \bar d_n\right) = L\left(P, a^*_{P, \mathcal{A}}\right)  + \int_Z \underbrace{\log\left(\frac{a^*_{P, \mathcal{A}}}{\bar d_n}\right)(z)}_{= \text{bias}_z}\mathrm{d}P(z) = L\left(P, a^*_{P, \mathcal{A}}\right) + \mathbb{E}_P\left[\text{bias}_z\right]. 

Note that in this case the higher order moments of the decision rule are
not zeros. We might approximate the volatility of the decision rule with
the second-order term of a Taylor expansion (see the appendix), but
relative entropy loss allows to use an alternative (exact) measure for
the variation in :math:`d`, the so called **Theil's second entropy**
[Theil-1967]_, which captures all higher order moments
of :math:`d`. We can derive it by writing

.. math::  R_n(P, d) - L(P, \bar d_n) = \mathbb{E}_P\left[\int_{Z^n} \log \left(\frac{p}{d(z^n)}\right)(z)\mathrm{d}P(z^n)\right]  - \mathbb{E}_P\left[\log\left(\frac{p}{\bar d_n}\right)(z)\right] = \mathbb{E}_P\left[ \underbrace{\left(\log \bar d_n - \mathbb{E}_{Z^n}[\log d(z^n)]\right)(z)}_{= \nu(d)_z}\right]. 

The volatility term indeed captures the variability of :math:`d(z^n)`
(as the sample varies). For example, :math:`\mathbb{V}[d(z^n)]=0`
implies :math:`\nu(d) = 0`. Furthermore, note that Theil's second
entropy measure of an arbitrary (integrable) random variable :math:`Y`
is

.. math:: \nu(Y) := \log \mathbb{E}Y - \mathbb{E}\log Y

This measure was utilized by [AlvarezJermann-2005]_ and [Backus-2014]_ in
the asset pricing literature. Essentially, it can be considered as a
generalization of variance or more precisely, both variance and
:math:`\nu` are special cases of the general measure of volatiliy

.. math:: f(\mathbb{E}Y) - \mathbb{E}f(Y), \quad\quad\text{where}\quad f'' < 0 .

The measure :math:`\nu` is obtained by setting :math:`f(y) = \log(y)`,
while the variance follows from :math:`f(y)=-y^2`.

Therefore, the excess risk of a decision rule :math:`d` under the
relative entropy loss is

.. math:: R_n(P, d) - L(P, a^*_{P, \mathcal{F}}) = \mathbb{E}_P\left[\text{misspecification}_z\right] + \mathbb{E}_P\left[\text{bias}_z\right] + \mathbb{E}_P\left[\nu(d)_z\right].

--------------

Classical approach -- the analogy principle
-------------------------------------------

In econometrics, classical approaches to estimation build heavily on the
empirical loss minimization principle, or as they often put it the
*analogy principle*. The underlying justification behind this
approach--that for simplicity we will call *classical*--is the *belief*
that the decision rule's induced finite sample distributions converge so
fast (with the sample size) that we can approximate them well with their
limiting distribution. For example, in lecture {asymptotic}, while
discussing the implied finite sample distributions of the MLE estimator
in the coin tossing example, we saw that the estimators distribution did
not change significantly after the sample size of 1,000. It has been
stressed several times before, this property of the decision rule
depends crucially on the complexity of the composite function class
:math:`\mathcal L_{\mathcal A}`.

**Remark:** Notice that in general there is no formal justification for
setting the finite sample distribution of the decision rule equal to the
limiting distribution. Typically, more accurate estimates can be
obtained via simulations, like Monte Carlo or bootstrap. We should add
in fairness that researchers of the classical approach often extend
their analysis with such techniques in order to assess the accuracy of
the large sample approximations.

.. raw:: html

   <!---
   Traditionally, the classical approach does not take into account the particular sample at hand while determining the decision problem's action space $\mathcal A$. In fact, since it relies almost exclusively on large sample approximations, it would not make much sense to do so. Later, when we turn to more modern approaches, we will see that this feature of the classical approach can cause serious troubles. 
   --->

In this notebook we denote the decision rules obtained by empirical loss
minimazation for a sample size :math:`n` as

.. math:: d^C(z^{n}):=\min_{a\in\mathcal A} \ L(P_n,a).

--------------

Some well-known examples are

-  **Non-linear Least Squares estimator:** In the case of regression
   function estimation, we can take

.. math::


   \begin{align*}
   &\mathcal A \subset \mathcal{F} = Y^X \quad\quad \text{and}\quad\quad L(P, \mu) = \int_{(Y,X)} (y - \mu(x))^2P(\mathrm{d}(y,x))\quad\quad\text{then} \\
   &\hspace{25mm}\widehat{\mu}^C(z^n) = \text{arg}\inf\limits_{\mu \in \mathcal{A}}\hspace{2mm}  \frac{1}{n}\sum_{t=1}^n (y_t - \mu(x_t))^2
   \end{align*}

-  **Maximum Likelihood Estimator:** In the case of density function
   estimation, we can take

.. math::


   \begin{align*}
   &\mathcal A \subset \mathcal{F} = \left\{f: Z \mapsto \mathbb{R}_+ : \int_Z f(z)\mathrm{d}z =  1 \right\}\quad\quad \text{and}\quad\quad L(p, f) = \int_{Z} p(z)\log \frac{p}{f}(z) \mathrm{d}z\quad\quad\text{then} \\
   &\hspace{35mm}\widehat{f}^C(z^{n}) = \text{arg}\inf\limits_{f \in \mathcal{A}}\hspace{2mm} - \frac{1}{n}\sum_{t=1}^n \log f(z_t) + \underbrace{H(p)}_{\text{entropy of }p}
   \end{align*}

-  **GMM estimator:** Having a set of moment restrictions and a positive
   semi-definite :math:`W`, we can take

.. math::


   \begin{align*}
   &\mathcal{A} = \{g(\cdot; \theta) : \theta \in \Theta\}\quad\quad \text{and}\quad\quad L(P, \theta) = \left[\int g(z; \theta)\mathrm{d}P(z)\right]' W \left[\int g(z; \theta)\mathrm{d}P(z)\right] \quad\quad\text{then} \\
   &\hspace{35mm}\widehat{\theta}_{W}(z^n) = \text{arg}\inf\limits_{\theta \in \mathcal{A}}\hspace{2mm} \left[\frac{1}{n}\sum_{t=1}^n g(z_t; \theta)\right]' W \left[\frac{1}{n}\sum_{t=1}^n g(z_t; \theta)\right]
   \end{align*}

--------------

We demonstrate key features of the classical approach by looking at the
specific example of vector autoregressions. This class of statistical
models, popularized in econometrics by [Sims-1980]_, are
extremely widely used tools in applied research. Although the simplicity
of linear models (due to their relatively low complexity) tends to mask
the generality of some of our findings, we believe this example is a
useful benchmark and provides sufficient intuition about the behaviour
of other (more complex) classical estimators.

VAR example
~~~~~~~~~~~

Let :math:`Z_t` denote an :math:`m`-dimensional vector containing the
values that the :math:`m` observable variables take at period :math:`t`.
Suppose that the statistician faces the following statistical decision
problem:

-  The assumed statistical models :math:`\mathcal H` are given by the
   class of ergodic covariance stationary processes over
   :math:`Z^{\infty}`.
-  The decision is based on the sample :math:`Z^{n}=\{Z_t\}_{t=1}^n`
   coming from observing :math:`Z_t` for :math:`n+k` time periods and
   conditioning on the first :math:`k` elements
-  The action space :math:`\mathcal{A}_k\subset \mathcal H` is equal to
   the set of :math:`k`-th order Gaussian vector autoregressions with

.. math::


   \begin{align*}
   &\hspace{3cm}Z_{t} = \mathbf{\mu} + \mathbf{A}_1Z_{t-1} + \mathbf{A}_2Z_{t-2} + \dots + \mathbf{A}_k Z_{t-k} + \varepsilon_{t}\quad\quad \varepsilon_{t}\sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})\quad \forall t\in\mathbb Z \\
   &\text{parameterized by }\quad\theta:= (\alpha, \sigma) \quad\quad\text{where}\quad \Pi := \left[\mu, \mathbf{A}_1, \dots, \mathbf{A}_k\right]',\quad \alpha: = \text{vec}\left(\Pi\right) \quad \text{and}\quad \sigma:= \text{vec}\left(\mathbf{\Sigma}\right)\quad\quad 
   \end{align*}

-  The loss function is relative entropy with the (conditional)
   log-likelihood function

.. math::


   \begin{align*}
   \log q_n(z^n \mid z_0, \dots, z_{-l+1}; \theta) & = -\frac{n}{2}\log(2\pi) + \frac{n}{2}\log\left|\mathbf{\Sigma}^{-1}\right| - \frac{1}{2}\sum_{t=1}^{n} \left(Z_{t} - \Pi' \tilde{Z}_{t}\right)'\Sigma^{-1}\left(Z_{t} - \Pi' \tilde{Z}_{t}\right) \\
   \text{where}&\quad \tilde{Z}_t:=\left[1, Z'_{t-1}, \dots, Z'_{t-k}\right]'
   \end{align*}

**REMARK:** It turns out that *for linear models*, model complexity is
closely related to the number of parameters used to define them.
[McDonald-2012]_ shows (see Corollary 6.4.) that the
class of VAR models with :math:`k` lags and :math:`m` time series has
Vapnik-Chernovenkis (VC) dimension :math:`km + 1`. VC dimension is an
alternative measure of complexity connected with Rademacher complexity.
In other words, when :math:`m` (or :math:`k`) is large, VAR models are
prone to overparametrization, or as [Sims-1980]_ puts it
they tend to be "profligately (as opposed to parsimoniously)
parametrized".

We are targeting the log density function of the data generating
mechanism with the assumption that it is

.. math:: \gamma(P_0) = \log p_0(z^n \mid z_0, \dots, z_{-l+1}) \in\mathcal{A}.

That is, for lag length :math:`k\geq l`, the misspecifcation error is
zero. We are working with a laboratory model where we know the truth and
generating syntetic samples for the analysis that follows. In order to
assign "realistic" values to the true :math:`\theta_0`, we use estimates
of a three-variate VAR fitted to quarterly US real GDP, real consumption
and real investment data over the period from :math:`1959` to
:math:`2009` using the dataset accessible through the
`StatsModels <http://statsmodels.sourceforge.net/devel/vector_ar.html>`__
database. See {REF notebook}.

For any prespecified lag length :math:`k`, the decision rules are given
by the maximum likelihood estimates

.. math:: \widehat{\Pi}^C(z^n) = \left[\sum_{t=1}^n \tilde{z}_t\tilde{z}'_t\right]^{-1}\left[\sum_{t=1}^n \tilde{z}_tz'_t\right] \quad\text{and}\quad \widehat{\Sigma}^C(z^n) = \frac{1}{n}\sum_{t=1}^{n} \left[z_{t} - \widehat{\Pi}'\tilde{z}_{t}\right]\left[z_{t} - \widehat{\Pi}'\tilde{z}_{t}\right]', 

with corresponding :math:`\widehat{\alpha}^C(z^n)`,
:math:`\widehat{\sigma}^C(z^n)` and :math:`\widehat{\theta}^C(z^n)`.

It is the distribution induced by the decision rule
:math:`\widehat{\alpha}^C(z^n)` on the action space :math:`\mathcal A_k`
that interests us for inference. In particular, we would like to know
how sensitive is the chosen action to the particular sample realization.
Controlling the true mechanism, we can get a very good sense about this
variation by using Monte Carlo methods. More precisely,

-  simulate :math:`R` alternative ensemble samples :math:`z_r^n`,
   :math:`r=\{1, \dots, R\}` from the true data generating mechanism
   VAR\ :math:`(l)`
-  fit a VAR\ :math:`(k)` model on each sample :math:`z_r^n` to obtain
   estimates :math:`\widehat{\alpha}^C_r`, for :math:`r=\{1, \dots, R\}`
-  take the standard deviation of the estimates (over different
   :math:`r`\ s) as the true volatility of
   :math:`\widehat{\alpha}^C(z^n)`

.. math:: s_n\left(\widehat{\alpha}^C\right) = \sqrt{\frac{1}{R}\sum_{r=1}^{R}\left(\widehat{\alpha}^C_r - \frac{1}{R}\sum_{i=1}^R\widehat{\alpha}^C_i\right)^2}

**REMARK:** Notice that the thus calculated "true volatility" in fact
provides the (parametric) bootstrap standard errors for the original
parameter estimates that we obtained with real data.

In the following figure, green (thin) lines represent alternative
samples that could have been generated by the true model. The used
sample is represented by the blue line. The horizontal green solid lines
denote the stationary means of :math:`Z_t`, the dashed lines are mean
:math:`\pm 2` stationary standard deviations.

.. figure:: ./alternative_samples.png
   :alt: 

Consistency
^^^^^^^^^^^

In practice, we can use only one sample (blue line). Nevertheless, from
ergodicity we know that if this sample is sufficiently large, then the
resulting sample statistics will provide good approximations to the
population counterparts. The decision rules at hand are plug-in
estimators, because their population versions are expressible as
analytic functions of the true moments of :math:`Z`. Consequently,
straightforward LLN argument implies that both
:math:`\widehat{\alpha}^C(z^n)` and :math:`\widehat{\Sigma}^C(z^n)` are
consistent.

(Note that here--in line with the classical literature--we deal with
convergence in the action space instead of convergence in the loss
space.)

Asymptotic standard errors
^^^^^^^^^^^^^^^^^^^^^^^^^^

However, consistency does not inform us about the variability of
:math:`\widehat{\alpha}^C(z^n)`. For that purpose the classical approach
brings to bear another powerful limit theorem: Central Limit Theory
(CLT). Under quite general regularity conditions (e.g. that all fourth
moments of :math:`Z^{\infty}` are finite), one can show that

.. math:: \sqrt{n}\left(\widehat{\alpha}^C(z^n) - \alpha\right) \overset{d}{\to} \mathcal{N}\left(0, \mathbf{\Sigma} \otimes  E[\tilde z_t\tilde z'_t]^{-1}\right)\quad\quad\text{as}\quad n\to \infty.

In other words, although the variation of the decision rule vanishes
asymptotically, if we multiply it by the scaling factor
:math:`\sqrt{n}`, the estimate :math:`\widehat{\alpha}^C(z^n)` will go
to :math:`\alpha` just at the rate so that a non-degenerate asymptotic
variation of :math:`\sqrt{n}\widehat{\alpha}^C(z^n)` is guaranteed. This
result is useful, because knowing

1. the asymptotic distribution: in this case normal
2. the asymptotic rate of convergence: in this case :math:`\sqrt{n}`

allows us to approximate the variation stemming from the finiteness of
the sample by using the asymptotic distribution and "scaling it back" by
the asymptotic rate. This steps can be summarized as

.. math:: \widehat{\alpha}^C(z^n) \approx \mathcal{N}\left(\alpha, \ AS_n\right)\quad\quad\text{where}\quad\quad AS_n := \frac{\mathbf{\Sigma} \otimes E[\tilde z_t\tilde z'_t]^{-1}}{n}

is the asymptotic covariance matrix of the decision rule
:math:`\widehat{\alpha}^C`. One drawback of this argument, however, is
that in practice we don't know the asymptotic covariance matrix, it
needs to be infered somehow form the sample :math:`z^n`. (Notice that
:math:`AS_n` does not depend on the sample, only on the sample size
:math:`n`.) One natural estimator is the sample analog of the asymptotic
covariance matrix, however, it is worth mentioning that there is no
clear reason why not to use another consistent estimator. By plugging in
an estimator for :math:`AS_n` at this step, we are inevitably
introducing an extra source of error on top of that :math:`\sqrt{n}` is
only an asymptotic rate of convergence not necessarily equal to the true
(finite sample) rate. Using the sample analog leads us to the so called
**approximate asymptotic standard error**:

.. math:: \widehat{AS}_n\left(\hat{\alpha}^C, z^n\right) := \sqrt{\frac{1}{n}\sum_{t=1}^{n} \left[z_{t} - \widehat{\Pi}'\tilde{z}_{t}\right]\left[z_{t} - \widehat{\Pi}'\tilde{z}_{t}\right]' \otimes \left[\sum_{t=1}^{n}\tilde z_t\tilde z'_t\right]^{-1}} \approx \sqrt{AS_n} \approx s_n\left(\widehat{\alpha}^C\right)

To get a sense of how well the true :math:`AS_n` and approximate
asymptotic standard error :math:`\widehat{AS}_n` really approximate the
true :math:`s_n`, the following figure reports their relative values:

-  the ratio between the true and the approximate asymptotic standard
   errors (black)
-  the ratio between the true and the asymptotic standard errors (green)

for alternative lag specifications when the true model has :math:`l=3`
and the sample size is :math:`n=100`.

.. figure:: ./relative_se1.png
   :alt: 

This figure forcefully reiterates our earlier insight that complexity of
the composite function class :math:`\mathcal L_{\mathcal{A}}`
(positively) affects the critical sample size above which large sample
approximations work well. The higher the complexity of the model
class--measured in terms of lag length--the worse the asymptotic
standard error in capturing the decision rule's finite sample variation.
In particular, using :math:`\widehat{AS}_n`, we tend to draw overly
optimistic conclusions about the estimator's variation and for a given
sample size this mistake is getting worse with the increase in model
complexity.

Adjusting for complexity
^^^^^^^^^^^^^^^^^^^^^^^^

Taking a closer look at the estimation error that the ignorance of the
true asymptotic covariance matrix implies, one can realize that the
:math:`n^{-1/2}` adjustment in fact "amplifies" that error in small
samples. We can write the approximate asymptotic standard error as

.. math:: \widehat{AS}_n\left(\hat{\alpha}^C, z^n\right) = \frac{n\widehat{\Sigma}(z^n) \otimes \left[\sum_{t=1}^{n}\tilde z_t\tilde z'_t\right]^{-1} - \mathbf{\Sigma} \otimes E[\tilde z_t\tilde z'_t]^{-1}}{n} + \frac{\mathbf{\Sigma} \otimes E[\tilde z_t\tilde z'_t]^{-1}}{n}

With small :math:`n`, the first term is multiplied by a relatively large
number. This suggests that by acknowledging the existence of estimation
error (first term), we might be able to adjust the scaling factor
:math:`n^{-1/2}` in a way to reduce this error. Moreover, because the
severity of the error hinges on model complexity (see the figure above),
it makes sense to use :math:`\tilde k:=1+km`, i.e. the
Vapnik-Chervonenkis dimension of the fitted VAR model, as an adjustment
factor. Influential papers in the classical literature [Sims-1980]_ often recommend to use
:math:`\left(n-\tilde k\right)^{-1/2}`, instead of :math:`n^{-1/2}` as a
scaling factor in order to "take into account the small-sample bias".
Clearly, an "alternative" interpretation of the augmented scaling factor
is to adjust the approximate asymptotic standard error for model
complexity.

.. math:: \widehat{AS}^{adj}_n\left(\hat{\alpha}^C, z^n\right) = \frac{n\widehat{\Sigma}(z^n) \otimes \left[\sum_{t=1}^{n}\tilde z_t\tilde z'_t\right]^{-1} - \mathbf{\Sigma} \otimes E[\tilde z_t\tilde z'_t]^{-1}}{n- \tilde k} + \frac{n}{n - \tilde k}\left(\frac{\mathbf{\Sigma} \otimes E[\tilde z_t\tilde z'_t]^{-1}}{n}\right)

Notice the subtle appearence of the ubiquitous bias-variance trade-off
in this expression. While the :math:`\tilde k`-adjustment is likely to
reduce the variation in the first term, it introduces bias for the
second term--which, of course, itself is just an approximation to the
true :math:`s_n\left(\widehat{\alpha}^C\right)`.

Indeed, as the following figure illustrates, the adjustment moves the
estimates in the "right" direction.

.. figure:: ./relative_se2.png
   :alt: 

**TODO** Misspecification? White robust correction

Efficiency
~~~~~~~~~~

As the previous sections demonstrate, an estimator's asymptotic
covariance matrix depends crucially on the action space (and loss
function) that the statistician entertains. This provides basis to rank
decision rules by invoking the criterion of *asymptotic efficiency*:
find the smallest possible asymptotic covariance matrix :math:`AS_n(d)`
relative to some well-defined class of estimators.

-  The famous Cramer-Rao bound provides a lower bound for the asymptotic
   covariance matrices of unbiased estimatrors. The correctly specified
   maximum likelihood estimator reaches this bound.
-  By choosing the weighting matrix :math:`W^*` appropriately, GMM
   estimators can be rendered efficient, i.e. it can be shown that
   :math:`W^*` leads to the lowest asymptotic covariance matrix whithin
   the class of GMM estimators with :math:`W\geq 0`.

Of course, this type of "minimization" of asymptotic covariance matrices
does not necessarily imply small finite sample variance of the decision
rule. The estimation-misspecification error decomposition helps
understanding the inherent trade-off.

Since the finite sample variance estimator :math:`\widehat{AS}_n` can be
linked with the volatility term of the estimation error, it might be
tempting to view the quest for asymptotic efficiency as a device to
minimize the estimaton error of the decision rule. Notice, however, that
this method does not take into account the possible *trade-off* between
the different moments of the decision rule. Instead, the classical
approach often restricts attention to unbiased estimators and looks for
the minimum variance estimator *among this class* (efficiency).
Nevertheless, the unbiasedness is gauged only relative to
the---restricted---range of the decision rule :math:`\mathcal{A}`. Even
if the decision rule is unbiased we still have misspecification error.
It seems difficult to defend the merits of an unbiased but misspecified
decision rule with large variance relative to a misspecified decision
rule with some bias and smaller variance. "Clever" estimators, like the
complexity-adjusted analog estimator of the asymptotic covariance matrix
above, trade-off bias/misspecification and variance flexibly taking into
consideration the available sample size and the complexity of the
estimation problem at hand.

--------------

From sample to population
~~~~~~~~~~~~~~~~~~~~~~~~~

The VAR example discussed above reveals the drawbacks of using
asymptotic theory to approximate a decision rule's finite sample
performance. A potential measure of discrepancy between the finite
sample and asymptotic behavior is the **estimation error**, which we
defined earlier as

.. math:: \mathcal{E}_d(P, \mathcal{A}, n) := R_n(P, d) - \inf_{a\in\mathcal{A}} \ L(P, a).

While the risk :math:`R_n(P, d)` captures the performance of :math:`d`
in samples of size :math:`n`, :math:`L(P, a^*_{P, L, \mathcal{A}})`
essentially encodes its asymptotic properties and from the consistency
of :math:`d` it follows that

.. math:: \lim_{n\to \infty} \ \mathcal{E}_d(P, \mathcal{A}, n) \overset{P}{=} 0.

In other words, even if :math:`d` is consistent relative to
:math:`(\mathcal{A}, \mathcal{H})`, its finite sample behavior still
hinges on the range of :math:`d`, i.e. the action space
:math:`\mathcal{A}`. Evidently, consistency implies that the estimation
error is not an issue in large samples, but without specifying the
sample size that counts large, this statement is mostly empty. The
notion of *large sample* is not absolute, it is always relative to the
complexity of the function class that we entertain.

Recall that the estimation error originates from the fact that we do not
know :math:`P`, instead we have to use the information in the (finite)
sample to approximate the 'best' action in :math:`\mathcal{A}`.
Intuitively, the smaller the estimation error the better this
approximation. Given a decision rule and a finite sample at hand we
would like to know how close the empirical loss and the true loss are.
Making sure that these two quantities are close to each other ensures
that the empirical loss is informative about the true loss. This
property is usually referred to as **generalization**.

Generalization
~~~~~~~~~~~~~~

Following [LuxburgSholkopf-2011]_ for a fixed
finite sample :math:`z^n`, we say that an assigned action
:math:`d(z^n)\in \mathcal{A}` *generalizes well*, if the quantity

.. math:: \left|L(P, d(z^n)) - L(P_n, d(z^n))\right|\quad \text{is small}.

Note that this property does *not* require that the empricial loss is
itself small, which is the objective function of the classical ELM
approach. It only requires that the empirical loss is close to the true
loss.

This sheds some light on what can go wrong with the ELM approach in
finite samples. In practice, one of the worst situations is
**overfitting**, that is, when the empirical loss is much smaller than
the true loss, hence our assessment of the quality of
:math:`d(z^n)\in\mathcal{A}` might be overly optimistic.

**Roadmap**

Note that the generalization property depends on the particular
realization of the sample. The realized sample determines the chosen
action, :math:`d(z^n)\in\mathcal{A}`, and the empirical distribution,
:math:`P_n`. In order to give statements regarding the generalization
property that extends to more than one paritcular realization of the
sample the following steps are taken:

-  In order to resolve the uncertainty about the chosen action,
   :math:`d(z^n)`, consider all the actions that are in the range of the
   decision rule, :math:`\mathcal{A}`.
-  In order to resolve the uncertainty about the empirical distribution
   either

   -  take expectations or
   -  characterize where the random variable concentrates via tail
      bounds.

      -  These are essentially equivalent.

-  Give statements which apply uniformly for data generating processes
   in a given class, :math:`P\in\mathcal{H}`.

Resolving variation across actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extending the generalization property to all actions in the range of
:math:`d`, :math:`a \in\mathcal{A}`, leads to the notion of
**generalization error**, defined as

.. math::  \Delta(P, z^n, \mathcal{A}) := \sup_{a\in\mathcal{A}} \ \left|L(P, a) - L(P_n, a)\right|.

When the loss functional takes the form
:math:`L(P, a) = \int l(a, z)\mathrm{d}P(z)` then the generalization
error is the supremum of a scaled empirical process indexed by the
function class :math:`\mathcal{L}_\mathcal{A}`. The finite sample
techniques discussed in {reference asymptotic notebook} prove to be
useful to characterize the behavior of
:math:`\Delta(P, z^n, \mathcal{A})`.

Resolving uncertainty about the empirical distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tail bounds**

To draw uniform inference about the generalization properties of
:math:`d`, we can use probabilistic tail bounds for
:math:`\Delta(P, z^n, \mathcal{A})`. One of the key defining element of
the tail bounds is the complexity of the class
:math:`\mathcal{L}_\mathcal{A}`.

We can apply {last theorem asymptotic notebook} in the current setting.
For uniformly bounded functions
:math:`\lvert l_a \rvert_{\infty} < B \ \forall l_a \in\mathcal{L}_\mathcal{A}`
for each :math:`\delta>0` we have that

.. math::  P \Big\{ \Vert P_n - P \Vert_{\mathcal{L}_\mathcal{A}} \geq  2\mathsf{R}\left(\mathcal{L}_{\mathcal{A}}, n\right)  + \delta \Big\} \leq 2 \exp\Big\{- \frac{n \delta^2}{2 B^2} \Big\}. 

**Average generalization error**

A somewhat less ambitious approach is to focus on the average
generalization error, i.e.

.. math:: \mathbb{E}_{Z^n}\left[ \Delta(P, z^n)\right] = \int_{Z^n} \sup_{a\in\mathcal{A}} \ \left|L(P, a) - L(P_n, a)\right| \mathrm{d}P(z^n).

Naturally, by bounding the tail probabilities of :math:`\Delta` we
control the mean as well. In fact ,with some technical care---using a
symmetrization argument---one can bound the expectation of the
generalization error using the Rademacher complexity of the class
:math:`\mathcal{L}_\mathcal{A}`,

.. math:: \mathbb{E}_{Z^n}\left[ \Delta(P, z^n)\right]\leq 2\mathbb{E}_{Z^n} \left[ \mathsf{R}\left(\mathcal{L}_{\mathcal{A}}(Z^n)\right) \right].

This inequality follows from a symmetrization argument discussed at the
end of {notebook 02}.

Unfortunately, the Rademacher complexity still depends on the unknown
distribution :math:`P` governing the iid sampling. There are many
different ways to bound the Rademacher complexity---and together the
expectation of the supremum of the empirical process.

The important message is that in order to control the generalization
property of the decision rule we need to limit the Rademacher complexity
of :math:`\mathcal{L}_\mathcal{A}`, which is the relevant measure of the
function class for the purposes of statistical inference.

Estimation error and generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It turns out that the average generalization error of the ELM decision
rule :math:`d^C` is tightly linked to its estimation error.

.. math:: \mathcal{E}_{d^C}(P, \mathcal{A}, n) =  \mathbb{E}_{Z^n}\Big[ L(P, d^C(z^n)) - L(P, a^*_{\mathcal{A}}) \Big]

Now, consider the following decomposition of the estimation error

.. math::


   \begin{align*}
   \mathcal{E}_{d^C}(P, \mathcal{A}, n) = & \mathbb{E}_{Z^n}\Big[ L(P, d^C(z^n)) - L(P_n, d^C(z^n))\Big] \\
   + & \underbrace{\mathbb{E}_{Z^n}\Big[ L(P_n, d^C(z^n)) - L(P_n, a^*_{\mathcal{A}}) \Big]}_{\leq 0} + \underbrace{\mathbb{E}_{Z^n}\Big[L(P_n,  a^*_{\mathcal{A}}) - L(P,a^*_{\mathcal{A}}) \Big]}_{= 0}\quad\quad  (1)
   \end{align*}

-  The second term on the RHS is nonpositive, because the decision rule
   is based on ELM, so :math:`L(P_n, d(z^n)) \ \leq \ L(P_n, a)` for all
   :math:`a\in\mathcal{A}`.
-  The last term disappears when we take the expectation, as we assumed
   that :math:`L` is linear in its first argument.

Hence, we have the following chain of ineqaulities

.. math:: \mathcal{E}_{d^C}(P, \mathcal{A}, n) \leq \mathbb{E}_{Z^n}\Big[ L(P, d^C(z^n)) - L(P_n, d^C(z^n))\Big] \leq \mathbb{E}_{Z^n}\Big[\sup_{a\in\mathcal{A}}\{L(P, a) - L(P_n, a)\} \Big].

The last term is equivalent to the above introduced average
generalization --- technically the expectation of the sup-norm of an
empirical process. We can use the techniques discussed in the {notebook}
to upper bound its value through notions of complexity.

This suggests that by seeking good generalization performance of the ELM
estimator the statistician can efficiently control the estimation error
as well, thus making sure that the asymptotic analysis provides a
relatively good approximation to the finite sample properties of
:math:`d^C`.

It is easy to see that one could always make the estimation error and
generalization error zero by choosing a constant decision rule -- that
is, one which range is a singleton and hence assigns the same action to
each possible realization of the sample. However, that decision rule
would ignore all the information that is available in the data -- the
source of information for statistical inference. The approach of
statistical learning theory attempts to balance this trade-off.

Statistical Learning Theory -- controlling complexity
-----------------------------------------------------

One criticism of the classical approach outlined in section {last
section} is that it does not deal with the generalization problem
arising in finite samples. Statistical learning theory takes a somewhat
different approach and attempts to balance good generalization and low
estimation error with small misspecification error.

Again, the objective is to minimize the excess risk of the decision
rule. As seen earlier the estimation-misspecification error
decomposition highlights one of the main dilemmas the statistician is
facing

.. math::  \underbrace{R_n(P, d) - L\left(P, a^{*}_{L, P, \mathcal{F}} \right)}_{\text{excess risk}} =  \underbrace{R_n(P, d) - L\left(P, a^{*}_{L, P, \mathcal{A}}\right)}_{\substack{\text{estimation error} \\ \text{random}}} + \underbrace{L\left(P, a^{*}_{L, P, \mathcal{A}}\right)- L\left(P, a^{*}_{L, P, \mathcal{F}} \right)}_{\substack{\text{misspecification error} \\ \text{deterministic}}}. 

-  The misspecification error captures the idea that the true feature of
   the DGP does not lie in the range of the decision rule, hence there
   is an inherent error due to this misspecification. Correspondingly,
   ceteris paribus enlarging the action space -- the range of the
   decision rule -- the misspecification error gets smaller. As
   :math:`\mathcal{A}` approaches :math:`\mathcal{F}` the
   misspecification error vanishes.

-  However, the range of the decision rule also plays a key role in the
   size of the estimation error and its ability to generalize. The
   non-asymptotic tail bounds teach us that in order to achieve low
   estimation error and good generalization the complexity of the class
   :math:`\mathcal{L}_\mathcal{A}` has to be small. The complexity is
   weakly increasing in the action space, :math:`\mathcal{A}`.

The above trade-off---inherent in all statistical inference
problems---can be visualized on the following graph.

.. figure:: ./decomp.png
   :alt: 

In terms of the action space of the decision rule,

-  whenever the gain from smaller misspecification error exceeds the
   loss from greater estimation error one should increase the action
   space, there is **underfitting**.
-  whenever the gain from smaller estimation error exceeds the loss from
   greater misspecification error one should decrease the action space,
   there is **overfitting**.

An ideal decision rule traces the minimum of the U shaped excess risk.
By controlling the range of the decision rule, the action space
:math:`\mathcal{A}`, the approach of statistical learning theory can
balance the trade-off between the estimation error and the
misspecification error.

Controlling excess risk through the action space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the previous discussion on how the size of the action space
affects both estimation and misspecification errors the modern
approaches to statistical inference explicitly control for complexity
through the decision rule's action space.

When the class of all admissible actions, :math:`\mathcal{F}`, is too
large for statistical inference the typical approach is to a priori
specify a nested sequence of action spaces
:math:`\{\mathcal{A}_k \}_{k \in \mathcal{K}}` whose union is equal to
:math:`\mathcal{F}`. That is
:math:`\mathcal{A}_k \subseteq \mathcal{A}_{k'}` whenever
:math:`k\leq k'` and :math:`\cup_{k}\mathcal{A}_k = \mathcal{F}`. The
problem of choosing an action space from this sequence is called *model
selection*. Formally, we would like to find the class which minimizes
the excess risk -- balancing the estimation and misspecification errors

.. math::  \min_{k\in\mathcal{K}}\Big\{ R_n(P, d_k) - L\left(P, a^{*}_{L, P, \mathcal{F}} \right) \Big\} =  \min_{k\in\mathcal{K}}\Big\{R_n(P, d_k) - L\left(P, a^{*}_{L, P, \mathcal{A}_k}\right) + L\left(P, a^{*}_{L, P, \mathcal{A}_k}\right)- L\left(P, a^{*}_{L, P, \mathcal{F}} \right) \Big\} 

where :math:`d_k` denotes the empirical loss minimizing decision rule
whose range is :math:`\mathcal{A}_k`.

The intuitive idea behind the approach -- often referred to as
*structural risk minimization* or *methods of sieves* -- is that one
should select action spaces inducing small complexity in smaller sample
sizes where the estimation error is more severe and as the sample size
grows select larger action spaces which ensures shrinking
misspecification error at least asymptotically.

This approach, of course, nests the classical frequentist one by setting
:math:`\mathcal{A}_k = \mathcal{A}\subseteq \mathcal{F}`,
:math:`\forall k\in \mathcal{K}`.

In this sense, we can think of the corresponding decision rule --
:math:`d^{SLT}` -- in terms of an indexed collection of action spaces
and a rule of selecting an element for each sample. Often, the selection
criterion is *data-dependent* as distribution-free upper bounds on the
complexity are usually too conservative.

Arguably, one of the most common and popular way of executing this
agenda is through penalization methods.

Penalized empirical loss minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Penalized empirical loss minimiziation has several forms, our aim here
is to highlight the conceptual similarities. In general, for each
potential sample one can consider the constrained optimization problem
of the following form

.. math::  d^{SLT}(z^n) :=  \arg \min_{a \in \mathcal{A}_{k^*}} \ L(P_n, a)

.. math::  \text{where} \quad k^* := \arg \inf_{k\in\mathcal{K}} \left\{ \min_{a \in \mathcal{A}_k} \ L(P_n, a) + \Phi(\mathcal{A}_k) \right\}.

:math:`\Phi` is a cost function penalizing the complexity of each
:math:`\mathcal{A}_k`. It is an assignment

.. math:: \Phi : \{\mathcal{A}_k\}_{k\in\mathcal{K}} \mapsto \mathbb{R}.

There are two logically distinct steps in this procedure.

1. The "inner loop" is the *empirical loss minimization* problem picking
   an action within a given model.
2. The "outer loop" is the *model selection* problem specifying the
   action space.

The model selection problem in the classical approach is "degenerate".
We can nest the classical decision rule corresponding to
:math:`\mathcal{A}` in the current framework as a special case of
:math:`\Phi` which

-  assigns zero to a prespecified element, :math:`\mathcal{A}`
-  and assigns infinity to every other element in the sequence,
   :math:`\{\mathcal{A}_k\}_{k\in\mathcal{K}} \setminus \mathcal{A}`.

Intuitively, for a given realization of the sample, the overall level of
the cost function :math:`\Phi` over its domain captures how much
emphasis we put on the estimation error versus the misspecification
error.

-  If the overall value level of the cost is low then the decision rule
   will likely pick an action from a large class for which the
   misspecification is less of an issue.
-  If the overall value level of the cost is high then the decision rule
   will likely pick an action from a small class for which the
   estimation error is typically not so severe.

It is important to note that both the sequence of action spaces and the
shape of the penalty term represents the statistician's prior knowledge
about the problem. They have a very similar role to that of the prior
distribution in Bayesian inference.

Operationalizing the cost function :math:`\Phi`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most often, the general penalty term is not written for a class of
actions but a single action. This will help us in rewriting the
constraint of the optimization problem. Define
:math:`\phi : \mathcal{F} \mapsto \mathbb{R}`

.. math::  \phi(a) := \inf_{k}\{\Phi(\mathcal{A}_k) : a \in \mathcal{A}_k\}.

We like to think of this definition as a projection of the action space
to the prespecified sequence of actions. Thus, the complexity penalty of
a single action corresponds to the complexity penalty of the first set
of actions in the sequence which contains the action in question.
Accordingly, we can characterize the action sets through the function
:math:`\phi`,

.. math:: \mathcal{A}_k \equiv \{a : \phi(a) \leq \Phi(\mathcal{A}_k)\}.

Frequently, the penalty term is defined through a norm in a reproducing
kernel Hilbert space (RKHS). Connections between these norms and
complexity measures are well-known in the literature. Having a penalty
term for each action we can recast the empirical loss minimization
problem as an unconstrained optimization problem over all actions,
:math:`\mathcal{F}`, via the method of Lagrange multipliers,

.. math::  d^{SLT}(z^n; \lambda) := \arg \min_{a\in\mathcal{F}} \ L(P_n, a) + \lambda (\phi(a) - \Phi(\mathcal{A}_k)).

The Lagrange multiplier :math:`\lambda` is corresponding to the
constraint :math:`\phi(a) \leq \Phi(\mathcal{A}_k)`. By changing the
Lagrange multiplier we can effectively control the "complexity radius"
of the constraint. In these cases :math:`\lambda` is called the *tuning
parameter* and it is the main tool in the model selection problem.

Model selection via cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As noted before selecting :math:`\lambda` effectively corresponds to
selecting a model -- how much complexity is the statisticion willing to
tolerate while trying to fit the data. We can treat it as a *taste
parameter* which describes the preferences of the decision maker (the
statistician in our case).

In practice however, as theoretical bounds on the complexity -- and
hence the penalty term -- are often too conservative, a popular way of
setting the tuning parameter and hence selecting a model is through
cross-validation. In this sense, the penalty term/cost function is
data-dependent, :math:`\Phi(\mathcal{A}_k, z^n)`.

The simplistic underlying idea is to get a direct estimate of the
selected action's performance through an independent sample --
information which is not used in the fitting phase. There is still a lot
of structure on the penalty term, however we wish to specify it's
overall level empirically and not theoretically.

Splitting the original sample randomly into two parts we have an
"in-sample" used for fitting and an "out-of-sample" used for testing.
Denote these by :math:`z^n = (z^n_{in}, z^n_{out})` and the
corresponding empirical measures as :math:`P_{in}` and :math:`P_{out}`.
A simplified version of cross-validation takes the following empirical
approach to model selection

.. math::  \lambda^* := \arg \inf_{\lambda} L\Big( P_{out},\ d^{SLT}(z^n_{in}; \lambda) \Big).

That is, loop over the values of :math:`\lambda` and

1. for each value of :math:`\lambda` pick an action based on the
   training sample

   .. math:: d^{SLT}(z^n_{in}; \lambda)

2. evaluate the performance of the selected action on the testing sample

   .. math:: L\Big( P_{out},\ d^{SLT}(z^n_{in}; \lambda) \Big)

3. choose the value of :math:`\lambda` for which the performance is best
4. finally, pick an action for the selected :math:`\lambda` based on the
   whole sample,

   .. math:: d^{SLT}(z^n; \lambda^*).

By choosing a certain specification of the defined components many of
the well known machine learning techniques can be treated simultaneously
in a common framework. The lasso and ridge regressions, support vector
machines or regularization networks can all be treated in the
above-defined setting.

Example -- OLS vs. Ridge
~~~~~~~~~~~~~~~~~~~~~~~~

We illustrate the idea of penalized empirical loss minimization and
model selection through a simple simulation of OLS regression and Ridge
regression for the same data set.

Assume that :math:`Z = (Y, X_1, X_2)` and we would like to estimate the
regression funciton predicting :math:`Y` as a function of
:math:`(X_1, X_2)`. The true regression function, :math:`\mu`, is
non-linear including a constant, level, quadratic and interaction terms.

Tha following graph shows the individual marginal effects of the
variables fixing other variables at their means.

.. figure:: finite_marg_effect.png
   :alt: 

We observe a random sample of size 100.

Suppose that we consider a model containing all level, quadratic and
interaction terms of the two covariates as we suspect ahead that the
relationship is non-linear---hence we are in a correctly sepcified
framework. The action space is relatively "complex" and hence it is
prone to fit any noise present in the data.

Define the non-linear feature mapping including all level, quadratic and
interaction terms together with a constant unit vector as
:math:`K(\mathbf{X})`.

The classical approach would be

.. math:: d^C(z^n) = \arg\min_{\beta} \sum_{i=1}^n \left(y_i - \langle \beta, K(x_i)\rangle\right)^2.

The Ridge approach -- a special case of the SLT approach -- for a given
tuning parameter :math:`\lambda` would be

.. math:: d^{SLT}(z^n; \lambda) = \arg\min_{\beta} \sum_{i=1}^n \left(y_i - \langle \beta, K(x_i)\rangle\right)^2 + \lVert\beta \rVert_2^2.

Knowing the true DGP we can simulate the true excess loss of the actions
picked by the different decision rules. We vary the tuning parameter of
the Ridge regression to trace out the model space and the corresponding
decision rules' performance.

.. figure:: ./finite_ridge_tuning.png
   :alt: 

As seen on the figure slightly shrinking the coefficients towards zero
helps to reduce the variance of the decision rule and improve
out-of-sample prediction. Ridge regression is equivalent to OLS when we
set the tuning parameter to zero and effectively cancel the penalty
term. For small values of :math:`\lambda` we reduce overfitting the
noise and hence smaller excess loss for the picked action relative to
OLS. However, for larger values the penalty term prevents us to pick up
the general characteristics of the data and the model underfits the
sample resulting in higher excess risk than that of the OLS.

--------------

References
----------

.. [AlvarezJermann-2005] Alvarez, Fernando & Jermann, Urban J. 2005. Using Asset Prices to Measure the Persistence of the Marginal Utility of Wealth. Econometrica, Econometric Society, vol. 73(6), pages 1977-2016, November.

.. [Abu-Mostafa-2012] Abu-Mostafa, Y. S., Magdon-Ismail, M., & Lin, H. T. (2012). Learning from data (Vol. 4). New York, NY, USA.

.. [Backus-2014] Backus, David & Chernov, Mikhail & Zin, Stanley. 2014. Sources of Entropy in Representative Agent Models. Journal of Finance, American Finance Association, vol. 69(1), pages 51-99, 02.

.. [LuxburgSholkopf-2011] Luxburg, U. von and Schölkopf, B. 2011. Statistical Learning Theory: Models, Concepts, and Results. In: D. Gabbay, S. Hartmann and J. Woods (Eds). Handbook of the History of Logic, vol 10, pp. 751-706.

.. [McDonald-2012] McDonald, Daniel J. (2012). Thesis: Generalization error bounds for state-space models. `Link <http://pages.iu.edu/~dajmcdon/research/dissertation/thesis.pdf>`__

.. [Sims-1980] Sims, C. A. (1980). Macroeconomics and reality. Econometrica: Journal of the Econometric Society, 1-48.

.. [Theil-1967] Theil, H. (1967). Economics and information theory. Amsterdam: North-Holland.

--------------

The code for the simulations and generating the graphs can be found `here <https://github.com/QuantEcon/econometrics/blob/master/Notebook_03_finite/finite_code.ipynb>`__.

--------------

Appendices
==========

(A) Taylor-expansion of the risk functional
-------------------------------------------

Remember that the risk of a decision rule :math:`d` is given by the
following expression.

.. math::  R_n(P, d) := \int\limits_{Z^n} L(P, d(z^n)) \ \mathrm{d} P(z^n) 

Consider the Taylor expansion of this functional with respect to the
decision rule around a particular :math:`d`. For any alternative
decision rule :math:`\tilde{d}`, we can define the difference

.. math:: \tilde{d} - d := \lambda \eta(z^n)\quad \quad \text{where}\quad \eta: \mathcal{S} \mapsto \mathcal{A}, \quad \lambda\in\mathbb{R}_+

and then the second-order Taylor expansion of the risk functional around
:math:`d` is

.. math::  R_n\left(P, \tilde{d}\right) = R_n\left(P, d \right) + \int_{Z^n} \frac{\partial L(P, d(z^n))}{\partial a}\lambda\eta(z^n)\mathrm{d} P(z^n) + \int_{Z^n} \frac{\partial^2 L(P, d(z^n))}{\partial a^2}\frac{\lambda^2\eta(z^n)^2}{2}\mathrm{d} P(z^n) + O(\lambda^{3})

where we use the notion of Gateaux differential (generalization of
directional derivate) to obtain the marginal change in the loss function
as the abstract :math:`a` changes.

An important reference point of any decision rule :math:`d` is the
expected action that it provides for a given sample size :math:`n`,

.. math:: \bar{d}_n := \int_{Z^n} d(z^n)\ \mathrm{d}P(z^n)

which does not necessarily belong to :math:`\mathcal{A}`. In what
follows, we imagine a decision rule :math:`\bar{d}_n\mathbf{1}(z^n)`
that assigns the value :math:`\bar{d}_n` to all sample realization
:math:`z^n` and use the Taylor approximation around this hypothetical
decision rule to approximate the risk of :math:`d`. In this case,
:math:`d - \bar{d}_n\mathbf{1} := \lambda \eta(z^n)` and

.. math::  R_n\left(P, d\right) = L\left(P, \bar d_n \right) + \int_{Z^n} \frac{\partial^2 L(P, \bar d_n)}{\partial a^2}\frac{(d - \bar{d}_n\mathbf{1})^2}{2}\mathrm{d} P(z^n) + O(\lambda^{3})

where the first-order term vanishes because the partial -- evaluated at
:math:`\bar d_n\mathbf{1}` -- is a constant and
:math:`\int_{Z^n}(d - \bar d_n\mathbf{1}) \mathrm{d}P = 0`. Note that in
this expression the second-order term encodes the theoretical variation
of the action that :math:`d` assigns to random samples of size
:math:`n`. The regular variance formula is altered by (one half of) the
second derivative of the loss function (evaluated at :math:`\bar d_n`),
representing the role of the loss functions's curvature in determining
the decision rule's volatility. As a result, a reasonable measure for
the decision rule's volatility can be defined as
:math:`R_n\left(P, d\right) - L\left(P, \bar d_n \right)`.

(B) Bias-variance-misspecification decomposition of GMM
-------------------------------------------------------

The elements of the problem are

-  *Observable:* :math:`Z \sim P`, with given moment conditions
   :math:`g: Z \times \mathbb{R}^{p+m} \mapsto \mathbb{R}^m`
-  *Action space:* :math:`\mathcal{A} = \Theta \subseteq \mathbb{R}^p`
-  *Admissible space:*
   :math:`\mathcal{F} = \Theta'\equiv \Theta \times \mathbb{R}^m`, so
   that we can always set the expectation of :math:`g` equal to zero by
   means of the :math:`m` auxiliary parameters.
-  *Loss function:*
   :math:`L(P, a) = \left(\int_Z g(z, a) \mathrm{d}P(z)\right)'W\left(\int_Z g(z, a) \mathrm{d}P(z)\right)`

Then the minimal loss is zero (by construction), i.e.
:math:`L(P, a^{*}_{P, \mathcal{F}}) = 0`.

The loss evaluated at the best-in-class action
:math:`a^*_{P, \mathcal{A}} = \inf_{a\in\mathcal{A}} \ L(P, a)`, is

.. math:: L\left(P, a^*_{P, \mathcal{A}}\right) = \mathbb{E}_P\left[ g\left(z, a^{*}_{P, \mathcal{A}}\right) \right]' W \mathbb{E}_P\left[ g\left(z, a^{*}_{P, \mathcal{A}}\right) \right] = \text{misspecification}

For the bias term we substract this quantity from the loss evaluated at
the average action
:math:`\bar d_n(z) := \int_{Z^n} a_{z^n} \mathrm{d}P(z^n)`

.. math:: L\left(P, \bar d_n\right) - L\left(P, a^*_{P, \mathcal{A}}\right) = \mathbb{E}_P\left[ g\left(z, \bar d_n \right) \right]' W \mathbb{E}_P\left[ g\left(z, \bar d_n \right) \right]  - \mathbb{E}_P\left[ g\left(z, a^{*}_{P, \mathcal{A}}\right) \right]' W \mathbb{E}_P\left[ g\left(z, a^{*}_{P, \mathcal{A}}\right) \right]  = \text{bias}

We approximate the volatility term with the second-order term of the
Taylor expansion. For simplicity, make use of the following notation

.. math:: D(a) := \mathbb{E}_P\left[ \frac{\partial g(z, a)}{\partial a}\right] \in \mathbb{R}^{m\times p} \quad\quad H(a) := \mathbb{E}_P\left[ \frac{\partial^2 g(z, a)}{\partial a^2}\right] \in \mathbb{R}^{p\times p\times m}

and so

.. math:: \frac{\partial L(P, a)}{\partial a} = 2 D(a)' W \mathbb{E}_P\left[ g(z, a)\right]\in \mathbb{R}^{p} \quad \quad \frac{\partial^2 L(P, a)}{\partial a^2} = 2 H(a) W \mathbb{E}_P\left[ g(z, a)\right] + 2 D(a)' W D(a) \in \mathbb{R}^{p\times p}

implying the approximation

.. math:: R_n(P, d) - L(P, \bar d_n)\approx \int_{Z^n} (d(z^n) - \bar d_n \mathbf{1}(z^n))'\left[ \underbrace{H(\bar d_n) W  g(z, \bar d_n)}_{\to 0 \ \text{as} \ n\to \infty} + D(\bar d_n)' W D(\bar d_n)\right](d(z^n) - \bar d_n \mathbf{1}(z^n)) \mathrm{d}P(z^n)

