---
layout: post
title: "[Notes] Isolation Forests"
excerpt: "Yet another deep-dive in finding a `catch-all` anomaly detection algorithm."
---

Twitter has some research on using a hybrid-STL model for "general" anomaly detection. It's "general" in that the model works on application and system metrics. I felt however, that it's still too involved. ISF (despite some of it's structural flaws) seems more promising and intuitive, while also being applicable to a wide array of datasets out-of-the-box.

## References
1) Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008. ICDM‘08 : https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
2) Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation-based anomaly detection.” ACM Transactions on Knowledge Discovery from Data (TKDD) : https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
3) S Guha, N Mishra, G Roy, O Schrijvers. "Robust Random Cut Forest Based Anomaly Detection On Streams" ICML 2016 : http://proceedings.mlr.press/v48/guha16.pdf

## 1) Isolation Forest (base paper) [ICDM 2008] https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

### Introduction
- Differs from standard practice of profiling normal samples and *then* identifying the anomalies. **This tries to isolate anomalies instead of profiling the normal points.**
- Major drawbacks for the standard(normal profiling) approaches:
    - Their detection is optimized to profile normal instances and NOT explicitly to identify anomalies. This could lead to FPs or FNs.
    - Might be constrained to low-dimensional data or high-dimensional data might be computationally expensive.

- Two exploits (hypothesis?) used by THIS paper:
    - Anomalies will be a minority class.
    - Attribute-values of anomalies will be very different from normal.

**Anomalies are few and different, and thus are more susceptible to isolation. iForest builds an ensemble of iTrees for a set of points. Anomalies are points which have short average-path lengths across iTrees.**

- Only two vars:
    - Number of trees to build.
    - Sub-sampling size.

- Goodness:
    - Only 2 vars.
    - Quick convergence (and high detection-performance) with small number of trees and small sampling size.
    - Since a large part of an iTree that isolates the normal points is not needed for anomaly detection, it does not need to be constructed. Smaller sample sizes produce better iTrees because the swamping and masking effects are reduced.
    - No distance or density measures are considered(IS THIS REALLY A PRO????)
    - Linear time complexity.
    - It can handle irrelevant attributes. WHAT AND HOW????


### Isolation and Isolation Trees
- Data-induced random-tree : partition till all instances are in a leaf.
- Anomalous points are likely to get partitioned quickly and easily. Normal points (by density or similarity) are more likely to be 'closer' and hence will have longer paths to separation.
- The ensemble of trees helps.

![](https://i.imgur.com/ZY2EESk.png)

### Isolation Tree
- $T$ is a node of an isolation tree. It is either a leaf-node, or an internal node containing one test and EXACTLY two daughter nodes ($T_l$, $T_r$).
- A test in an internal node consists of an attribute $q$ and a split value $p$ such that the test $q<p$ divides the data points into $T_l$ and $T_r$.
- Data is $X = {x_1, x_2, ..., x_n}$ of $n$ samples/instances from a $d$-variate distribution.
- Procedure to build iTree:
    - Recursively divide $X$ by randomly selecting an attribute $q$ and a split value $p$ until termination cases:
        - Tree reaches height limit, or
        - $|X| = 1$, or
        - All data in $X$ has the same values.

    - If all instances are unique, then every instance will have a leaf node.
    - Final tree should have $n$ leaves and $n-1$ internal nodes. Total nodes will be $2n - 1$.

- $h(x)$ is path length of a point. It is the number of edges that instance $x$ traverses in an iTree.
- *AnomalyScore* is calculated using the "expected length of $h(x)$" from a collection of trees == $E(h(x))$.
    - $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$
    - $E(h(x)) =$ average length of $h(x)$ across trees.
    - $c(n)$ has a formula.
    - when $E(h(x)) == c(n)$ then $s == 0.5$ 
    - when $E(h(x)) == 0$ then $s == 1$
    - and when $E(h(x)) == (n−1)$ then $s == 0$