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

> I am choosing to convert everything to lower-case, and also ignore the punctuation. These are probaly valuable features, but we can do without them for this implementation.

**Fun-fact**: You know that "Penn TreeBank" which keeps popping up everywhere in NLP literature? I just assumed that a dataset so oft-cited *must* be free. Turns out it costs \\$1700 for non-members, and \$850 for a "reduced license"...

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