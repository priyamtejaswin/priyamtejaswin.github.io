---
layout: post
title: "The Structured Perceptron for Structured Prediction"
excerpt: "Or, realising just how painfully slow Python can get."
---

**Structured Prediction** (or Structured Learning) deals with learning algorithms where the ouput is not a single value, but rather a *structured* object: sequence, tree, etc. Consider the task of [Part-Of-Speech Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) where the goal is to tag every word with its appropriate part of speech: verb, noun, adjective, etc. In the following sentence (picked from the Wiki link)

$$
\text{The sailor dogs the hatch.}
$$

the word $\text{"dogs"}$ could be mapped to a noun or a verb. A naive token classifer that only looks at the input tokens will ignore dependencies between the current tag and the previous tags. Structured Prediction builds on this. The features to predict the next tag will not only include the input tokens, but also the previously predicted tokens. In this sense, the features can be defined over the entire input and tags.

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

I use `get_clean_line` to generate `word/tag` sequences. This function reads a line, stores the tokens and their respective tags in separate lists, and returns them.
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
                return False

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
    # Uni-gram features: check for occurrence of a token with a label.
0: the_DT
1: small_JJ
2: dog_N
3: is_BBB
4: barking_VBG
    # Bi-gram features: check for occurrence of token pairs with a label.
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
&= \text{argmax}_{y_i}\ \phi(x_{1:i}, y_i).\bar{\alpha}
\end{align}
$$

To train this classifier, we loop over each training sample, predicting the label $\hat{y}_i$ and comparing it to the training label $y_i$. If these match, it means our current paramter values for $\bar{\alpha}$ *suffice* and need not be updated. If they don't, then we update. For each element $s \leq n$ in the vector,

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
Now, instead of predicting the most probable label for each token, we'll try to generate the most likely *sequence* of POS-tags.

$$
\begin{align}
\hat{y} &= \text{argmax}_{y \in \text{GEN}(x)}\ \phi(x, y).\bar{\alpha}
\end{align}
$$

This looks quite similar, but there are some differences.

$\phi$ **maps squences to a feature vector.**

The function $\phi$ is a joint function that maps input tokens $x$ *and* the labels *y* to a fixed length vector. It accepts the entire input sequence and a candidate label sequence. Our function $f$ is now returning a score for a pair of sequences.

$$
f(x, y) = \phi(x, y) . \bar{\alpha}
$$

Expanding the dot product, we observe

$$
f(x, y) = \phi(x, y) = \sum_{i}^t \sum_{s}^n \bar{\alpha}_s . \phi_s(h_i, y_i)
$$

Collins' paper defines $h_i$ to be the "context history" with which the prediction is made. It's defined as the tuple $(y_{-2}, y_{-1}, x_{1:n}, i)$, where $y_{-2}, y_{-1}$ are the tags for the last two tokens and $x_{1:n}$ is the trail of previous tokens. The paper also defines a global feature function $\Phi$

$$
\Phi_s(x, y) = \sum_i^t \phi_s(h_i, y_i)
$$

Since all features are binary, $\Phi_s$ simply counts all occurrences of the "local" feature $\phi_s$ in the sequence. With this defined, the scoring function $f(x, y)$ becomes:

$$
\begin{align}
f(x, y) = \phi(x, y) . \bar{\alpha} &= \sum_{i}^t \sum_{s}^n \bar{\alpha}_s . \phi_s(h_i, y_i) \\
&= \sum_s^n \bar{\alpha}_s . \Phi_s(x, y)
\end{align}
$$

The parameter update procedure remains the same; we swap the local feature, with the global $\Phi$ counts

$$
\begin{align}
\bar{\alpha}_s  &= \bar{\alpha}_s + \Phi_s(x, y) - \Phi_s(x, \hat{y}) \\ \\
& \text{Where} \\
& y  \text{ is the gold label sequence} \\
& \hat{y}_i \text{ is the candidate label sequence}
\end{align}
$$

All that's left is to find the highest scoring sequence ... which brings us to the other thing.

**Candidate** $\hat{y}$ **is an entire sequence.**

We consider the prediction $\hat{y}$ over the *sequence* of possible labels for all tokens in $x$ (captured in $\text{GEN}(x)$); as opposed to all possible labels for a single token. For a sequence of $N$ tokens with $S$ possible labels, search space over $\hat{y}_i$ was $N \times S$ if we were predicting labels independently for each token. Here, exhaustive search will be of the order $S^N$!

## Viterbi Decoding
The process of generating possible tag sequences is also called decoding. Exhaustive search is infeasible. A better option would be **Greedy Decoding**: you select the most probable label at for the current timestep, *fix it*, and then move on to the next timestep. Collins' paper uses the [**Viterbi Algorithm**](https://en.wikipedia.org/wiki/Viterbi_algorithm). If you're familiar with dynamic programming then you should refer to the pseudo-code in the Wiki link. In a separate exercise, I spent some time on the Viterbi Algorithm in the context of HMMs. I'm borrowing from that here.

To motivate why Viterbi works, and it's link to Dynamic Programming, consider a simpler problem you know the best tags upto timestep $t-1$. Now, you simply have to pick the tag that maximizes the assignment for the final token. We can extend this logic back to the first token. In the figure below, $x$ is the observed variable (i.e. the **token**) and $z$ is the unknown state variable (i.e. the **tag** $y_t$). Consider the state space to be $3$ (i.e. we only have 3 tags to select from) and our history tuple to be $(y_{t-1}, x_{[1:t]}, y_t)$.

<div class="post-image">
<img src="/assets/images/hmm-viterbi.svg">
<p><em><font size="-1">Viterbi state selection.</font></em></p>
</div>

For the first observation $x_1$ there is only one state variable, $z_1$. Since $y_0$ is `None`, we can assign a score to all three states using the parameter vector $\bar{\alpha}$ and considering the features for $x_1 \times y_1$ (i.e. combinations of $x_1$ with the three possible states). These will serve as the scores for $s_1, s_2, s_3$. For the next observation $x_2$, we have to consider all possible assignments for the previous token -- remember that our history for $x_2$ contains $y_2$ *and* $y_1$. Thus, to compute the scores $s$ for $z_2$, we *consider* all possible previous assignments ($z_1$), and *select* the link that maximizes the score for $z_2$.

In the figure, this process is represented by the colours of the links. For each token value $s$ in $z_2$, we consider all three tag assignments *upto* the previous token -- this is captured in the $s$ scores for $z_1$. For each state $s$ in $z_2$, after considering all three paths, we select the one which maximizes the score for that state, and update the score. In the [code](https://github.com/priyamtejaswin/c00lHaX/blob/master/ner_perceptron/collins_perceptron.py#L348), this process is called `Phase 1`. The time complexity for this phase is $N \times S^2$.

To generate the final sequence, we backtrack from the last timestep, selecting the most probable state assignments, through the most likely paths.

# The training loop
This is my main function -- the comments explain everything.
```python
@plac.annotations(
    path_brown_corpus = ("Path to Brown Corpus dir.", 'positional', None, str)
)
def main(path_brown_corpus):
    """
    Runs Collins' averaged perceptron tagging algorithm.
    """
    print "Files dir:", path_brown_corpus

    if not os.path.isdir(path_brown_corpus):
        raise OSError("path_brown_corpus -- %s does not exist."%path_brown_corpus)

    # Some code for loading the data ...
    files = [os.path.join(path_brown_corpus, name) for name in 
             os.listdir(path_brown_corpus) if len(name) == 4 and name[0] == 'c']
    data = load_files(files[:50])
    random.shuffle(data)

    # Train/test split ...
    tsplit = int(0.9 * len(data))
    train_data, test_data = data[:tsplit], data[tsplit:]
    print "Train seqs:", len(train_data)
    print "Test  seqs:", len(test_data)

    # Extract the bi-gram, uni-gram features for every (words, tokens) pair.
    train_feats = [get_features(w, t) for w,t in tqdm(train_data)]
    # `get_words_tags_weights` initializes the weights (wObs, wTags) and
    # creates the index maps for tokens, tags.
    word2ix, ix2word, tag2ix, ix2tag, wObs, wTags = get_words_tags_weights(train_feats)

    train_loop(train_data, word2ix, tag2ix, ix2tag, wObs, wTags, test_data)
```

And the `train_loop` looks something like this
```python
for outer in range(5):  # Or *epochs* if you're familiar with neural networks.
    for wdseq, tgseq in tqdm(train_data):
        counter += 1
        if counter % 500 == 0:
            # Code for computing stats on the test data.
            ...

        # Generate the most likely tag sequences with the current parameters.
        tr_gold, tr_pred = forward(wdseq, tgseq, word2ix, ix2tag, wObs, wTags)
        # The `forward` function returns the global features for the gold sequence 
        # and the candidate sequence.
        # These are passed to the `train_step` to update parameters.
        obs_pos, tag_pos = train_step(tr_gold, tr_pred, word2ix, tag2ix)
        for (r, c), v in obs_pos.items():
            if v != 0:
                wObs[r, c] += v  # `v` is the *net* change in the parameter value.

        for (r, c), v in tag_pos.items():
            if v != 0:
                wTags[r, c] += v
```

# Closing thoughts
Despite being  extremely simple, the Perceptron model can be a formidable baseline. There's an [excellent post by Matthew Honnibal](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python) (founder and creator of [spaCy](https://spacy.io/)) where he writes an extremely accurate POS-Tagger in less than 200 lines. This achieves an accuracy of 97%.

While I did not implement the full feature-set, I was curious to know how accurate my implementation was. But I was never really able to train it on the full dataset because of the decoder. It is *ridiculously* slow. 
I could have made some performance improvements (and Honnibal covers some in his other posts). Also, some computation can also be trivially parallelized. For instance, you can decode each instance in the batch separately in a different thread, or using matrices if you write the code correctly. But even then, the $N \times S^2$ time complexity of the decoder will slow you down, especially when you start working with a huge number of tags. I could re-write the entire thing in C, but maybe I don't have to. After all, Spacy is in Python, and it is one of the fastest libraries out there.

The trick (as championed by Spacy) is to implement all the bottlenecks in [Cython](https://cython.org/), a static compiler for Python code -- not to be confused with [CPython](https://github.com/python/cpython), which is the Python reference implementation you are probably using. But I'll leave that for a different post.

#### Some important stuff I skipped 
* The feature set that the paper uses is listed in [Ratnaparkhi, 1996](https://www.aclweb.org/anthology/W96-0213.pdf).
* The parameter update should be averaged across a batch. This is discussed in detail in Sections 2.5 and 3.
* I should be measuring F-measure, but I was too lazy.

Feel free to raise PRs if you spot something's off!