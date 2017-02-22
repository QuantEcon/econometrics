
Econometrics & Statistics for QuantEcon
=======================================

**February 2017**

Introduction
------------

The aim of these notebooks is to present concepts and techniques common
in modern statistical analysis to an audience familiar with
econometrics. Along the way we attempt to provide examples of estimation
problems well-known in econometrics to illuminate the introduced
concepts.

Notebook-01 -- Estimators as Statistical Decision Functions
-----------------------------------------------------------

Notebook-01 presents the key objects and concepts of econometrics and
statistics within the framework of statistical decision theory. We find
the framework useful for organizing our thoughts and comparing different
approaches to similar statistical problems thus highlighting the key
differences and similarities. We do not take the framework as a starting
point strictly defining optimality.

The introduced framework lets us separate assumptions into three major
groups.

    Assumptions on **statistical models** describing the data generating
    process. Typically, these assumption serve as an anchor defining a
    class of distirbutions relative to which we would like to establish
    certain properties. An estimator can be consistent or unbiased
    relative to one set of distributions but not relative to another.

    Assumptions on the **action space** restrict the set from which we
    allow a decision rule to take values from. As we will see, one of
    the key tools to control for certain statistical properties of
    decision rules is through controlling the action space.

    Assumptions on the **loss function** define which features of the
    data generating process we are targeting and specify the manner in
    which we are punishing different approximations to this feature.

Notebook-01 then presents statistical decision rules and characterizes
them through the distributions they induce on the action space and the
reals through the loss function. We set the **risk** of a decision rule
as the benchmark for comparison.

Finally, with the introduced concepts we discuss the important
**estimation error-misspecification error** decomposition of decision
rules.

Notebook-02 -- Asymptotic Analysis and Consistency
--------------------------------------------------

Notebook-02 takes an asymptotic approach to analyze decision rules. We
discuss conditions under which **consistency** holds and investigate the
determinants of **rate of convergence**.

For this end, Notebook-02 covers the basics of **concentration
inequalities**. We make connections between these inequalities and
measures of complexity with a special emphasis on empirical processes.
We introduce the concept of **Rademacher-complexity** and
Vapnik-Chervonenkis dimension to capture the size of a function class
for the purpose of statistical inference.

As it turns out the introduced concepts provide deep insights about the
asymptotic behavior of decision rules. Both consistency and the rate of
convergence crucially depends on the complexity of the action space.

Notebook-03 -- Coping with Finite Samples
-----------------------------------------

Notebook-03 explores the issues arising from dealing with finite samples
relative to an asymptotic approximation. We further decompose the risk
into **bias**, **volatility** and **misspecification** terms.

We discuss **generalization** properties of decision rules and their
connection to the estimation error. We apply the **non-asymptotic tail
bounds** covered in Notebook-02 to get a grasp on the finite sample
performance of decision rules.

The notebook discusses two approaches to estimation and the
corresponding philosphies underlying them. One is a frequentist approach
labelled as **classical** which is more common in the econometrics
literature. The other which we label as the approach of **statistical
learning theory** underlies many of the techniques common in modern
statistics.

Notebook-04 -- The Bayesian Approach (to be added)
--------------------------------------------------

Notebook-04 would introduce the Bayesian approach to statistical
inference. Tentative sections:

-  Conditions for consistency with special focus on the complexity of
   the prior.
-  Rate of convergence through capacity control on the prior
-  Implications of misspecification
-  Interpretation as a regularization technique

