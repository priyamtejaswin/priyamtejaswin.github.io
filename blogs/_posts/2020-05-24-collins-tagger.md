---
layout: post
title: "The Structured Perceptron for Structured Prediction"
excerpt: "Or, realising just how painfully slow Python can get."
---

**Structured Prediction** (or Structured Learning) deals with learning algorithms where the ouput is not a single value, but rather a *structured* object: sequence, tree, etc. Consider the task of [Part-Of-Speech Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) where the goal is to tag every word with its appropriate part of speech: verb, noun, adjective, etc. In the following sentence (picked from the Wiki link)

$$
\text{The sailor dogs the hatch.}
$$

the word $\text{"dogs"}$ could be mapped to a noun or a verb. A naive (singular)token classifer, though aware of the input tokens, will ignore dependencies between the current tag and the previous tags. Structured Prediction builds on this. The features to predict the next tag will not only include the input tokens, but also the previously predicted tokens. In this sense, the features can be defined over the entire input and tags.

The specific algorithm that we'll explore is presented in [Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms](https://www.aclweb.org/anthology/W02-1001.pdf), published by [Michael Collins](http://www.cs.columbia.edu/~mcollins/) in EMNLP 2002. This paper was awarded the [ACL Test-of-Time Award in 2018](https://naacl2018.wordpress.com/2018/03/22/test-of-time-award-papers/).

# Data
```bash
1. Write about specific problem.
2. Link to data, describe it.
3. Tid-bit about the PennTree bank, and how expensive it is.
4. Some notation about the data.
```

# Collins' Perceptron
```bash
1. Start by describing the simple perceptron.
2. Modify to Collins Perceptron.
3. Add code nippets if/where appropriate.
```

# Code

# HMM training
```bash
1. Mention complicated derivation.
2. Explain this as alternative for training.
```

# Comment on evaluation/metrics
```bash
1. How do we measure goodness.
2. Why is this so slow.
3. End.
```