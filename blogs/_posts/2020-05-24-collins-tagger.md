---
layout: post
title: "The Structured Perceptron for Structured Prediction"
excerpt: "Or realising just how painfully slow Python can get."
---

**Structured Prediction** (or Structured Learning) deals with learning algorithms where the ouput is not a single value, but rather a *structured* object: sequence, tree, etc. Consider the task of [Part-Of-Speech Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) where the goal is to tag every word with its appropriate part of speech: verb, noun, adjective, etc. In the following sentence (picked from the Wiki link)
$$
\text{The sailor dogs the hatch.}
$$
the word $\text{"dogs"}$ could be mapped to a noun or a verb. A naive (singular)token classifer, though aware of the input tokens, will ignore dependencies between the current tag and the previous tags. 