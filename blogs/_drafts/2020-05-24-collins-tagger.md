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
The task at hand is [Part-Of-Speech Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging). Every input token ($\text{The}$) is tagged with its appropriate part-of-speech ($\text{DT/Determiner}$). Part-of-speech tags are essential input features for many statistical NLP models.

Such "paired" data is our input. The task is to learn a model that can accurately predict the tags for a new sequence of words.

The specific dataset we'll use here is the [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus). You can download and extract the data by running the following
```bash
wget https://archive.org/download/BrownCorpus/brown.zip -O brown.zip
unzip brown.zip
```
or, if you have the GitHub code, by running `make data` from the source folder. Inside, you'll find a collection of text files with a `c[GENRE][NUMBER]` format. The `CONTENTS` file lists the genres and the text references used in the files. On inspecting a file, 
```


	The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn of/in Atlanta's/np$ recent/jj primary/nn election/nn produced/vbd ``/`` no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./.


	The/at jury/nn further/rbr said/vbd in/in term-end/nn presentments/nns that/cs the/at City/nn-tl Executive/jj-tl Committee/nn-tl ,/, which/wdt had/hvd over-all/jj charge/nn of/in the/at election/nn ,/, ``/`` deserves/vbz the/at praise/nn and/cc thanks/nns of/in the/at City/nn-tl of/in-tl Atlanta/np-tl ''/'' for/in the/at manner/nn in/in which/wdt the/at election/nn was/bedz conducted/vbn ./.


	The/at September-October/np term/nn jury/nn had/hvd been/ben charged/vbn by/in Fulton/np-tl Superior/jj-tl Court/nn-tl Judge/nn-tl Durwood/np Pye/np to/to investigate/vb reports/nns of/in possible/jj ``/`` irregularities/nns ''/'' in/in the/at hard-fought/jj primary/nn which/wdt was/bedz won/vbn by/in Mayor-nominate/nn-tl Ivan/np Allen/np Jr./np ./.

```
you'll find sentences are split into separate lines. Each token is accompanied with a POS-tag separated by a `/` -- `<token>/<tag>`. You can find tag abbreviations and definitions on the [Brown Corpus Wiki page](https://en.wikipedia.org/wiki/Brown_Corpus). Punctuation characters do not have a tag -- the character is repeated after `/`.

```python
def get_clean_line(line):
    line = line.lower().strip()
    if len(line) < 1:
        return None
    else:
        words, tags = [], []

        for token in line.split():
            try:
                word, tag = token.strip().split('/')
            except:
                return False  # Is this allowed?

            if word == tag:  # Ignore.
                pass
            elif tag == '.':  # Sentence closer.
                pass
            else:
                words.append(word)
                tags.append(tag.upper())

        if len(words) < 1:
            return None
        else:
            return words, tags
```

I use `def get_clean_line` to generate `word/tag` sequences. This function reads a line, stores the tokens and their respective tags in separate lists, and returns them.

> I am converting everything to lower-case, and also ignoring the punctuation. These are probaly valuable features, but we can do without them for this implementation.

**Fun-fact**: You know that "Penn TreeBank" which keeps popping up everywhere in NLP literature? I just assumed that a dataset so oft-cited *must* be free. Turns out it costs \\$1700 for non-members, and \$850 for a "reduced license"...

# Collins' Perceptron
Before the **Structured Perceptron**, let's have a look at the general Perceptron Algorithm.

## Perceptron Algorithm
Our training samples are pairs of sequences $(x, y)$. Consider a different task where we predict a label $\hat{y_i}$ for input token $x_i$. We define a linear function $f(x_i, y_i)$ as 

$$
f(x_i, y_i) = \phi(x_{0:i}, y_i).\bar{\alpha}
$$

$\phi$ returns a feature vector for an input sequence $x_{0:i}$, and a candidate label $y_i$. Usually, this is a binary feature generator. It accepts all input token leading up to $x_i$ -- you can also use tokens appearing ahead of $x_i$. This sequence of input tokens allows us to have bi-gram and tri-gram features associated with training class labels. Consider the following sequence of $\text{token/{label}}$ pairs:

$$
\text{<start> The/{DT} small/{JJ} dog/{N} is/{BBB} barking/{VBG} <end>}
$$

The full set of labels is available in the Brown Corpus. For the moment, assume our vocabulary of tokens and labels is limited to this sentence (along with `<start>` and `<end>`). One could define the feature vector of size $n$ in the following manner:

```python
[
    # Uni-gram features: check for occurrance of a token with a label.
0: the_DT
1: small_JJ
2: dog_N
3: is_BBB
4: barking_VBG
    # Bi-gram features: check for occurrance of token pairs with a label.
5: <start>_the_DT
6: the_small_JJ
7: small_dog_N
8: dog_is_BBB
# ...
]
```
With this, the feature vector for the first token ($\text{the}$) will have $1$ in positions 0 and 5, with the rest being zero.

$\bar{\alpha}$ is the paramater vector for our features -- it is also of size $n$. Dot-product of these vectors returns a score. Since the input $x_i$ is fixed, the predicted label, $\hat{y}_i$, will be the candidate label which gets the highest score.

$$
\begin{align}
\hat{y}_i &= \text{argmax}_{y_i}\ f(x_i, y_i) \\
&= \text{argmax}_{y_i}\ \phi(x_{0:i}, y_i).\bar{\alpha}
\end{align}
$$

To train this classifier, we loop over each training sample, predicting the label $\hat{y}_i$ and comparing it to the training label $y_i$. If these match, it means our current paramter values for $\bar{\alpha}$ *suffice* and need not be updated. If they don't, then we update. For each element $s < n$ in the vector,

$$
\begin{align}
\bar{\alpha}_s  &= \bar{\alpha}_s + \phi_s(x_{0:i}, y_i) - \phi_s(x_{0:i}, \hat{y}_i) \\ \\
& \text{Where} \\
& y_i  \text{ is the gold label} \\
& \hat{y}_i \text{ is the predicted label}
\end{align}
$$

If $\hat{y}_i$ and $y_i$ match, then the weights are not updated. Else, they are increased/decreased appriopriately (+1/-1).


## Structured Prediction
Now, instead of predicting the most probable label, we'll try to generate the most likely *sequence* of POS-tags.

$$
\begin{align}
\hat{y} &= \text{argmax}_{y \in \text{GEN}(x)}\ \phi(x, y).\bar{\alpha}
\end{align}
$$

This looks quite similar, but there are two differences:
1. The function $\phi$ is a joint function that maps input tokens $x$ *and* the labels *y* to a fixed length vector.
2. We consider the prediction $\hat{y}$ over the *sequence* of possible labels for all tokens in $x$ (captured in $\text{GEN}(x)$); as opposed to all possible labels for a single token.

The 2nd point is critical: consider a sequence of $N$ tokens with $S$ possible labels. In the previous formulation, search space over $\hat{y}$ was simple $N \times S$. Here, exhaustive search will be of the order $S^N$!

## Viterbi Decoding
The process of generating possible tag sequences is also called decoding. Exhaustiv search is infeasible. A better option would be **Greedy Decoding**: you select the most probable label at for the current timestep, *fix it*, and then move on to the next timestep.


# Comment on evaluation/metrics
```bash
1. How do we measure goodness.
2. Why is this so slow.
3. End.
```