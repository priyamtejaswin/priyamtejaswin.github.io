---
layout: post
title: "[Notes] Attention in Vision."
excerpt: "Trying to understand attention-based vision systems."
---

In the DeViSE project, we wanted to achieve something similar to what these papers explore. Except we took a completely different route (the Github README should explain things).

This post briefly covers the following papers on "attention-based" vision systems.
- base InfoGan paper (2016 - Chen, Duan and OpenAI folks) : https://arxiv.org/pdf/1606.03657.pdf
- Understanding mutual-info in context of InfoGan (2016 - Evidoma, Drozdov; NYU folks) : https://kevtimova.github.io/docs/Drozdov_Evtimova.pdf

## Paper reviews

### Understanding mutual-info in context of InfoGan (2016 - Evidoma, Drozdov; NYU folks) : https://kevtimova.github.io/docs/Drozdov_Evtimova.pdf

Learning to generate images with task-specific codes.

**or**

(*how to make InfoGans actually do something useful rather than returning a lot of codes and then hunting for which ones make the best plots in the paper*)

- InfoGan ties the output of the generator to a component of its input (i.e. *latent codes*).
- Autoencoders treat the input as target and work a reconstruction loss. VAEs can work with latent states.
- Mode collapse is a problem with GANs. If data is MNIST, then there is no explicit component/signal which enforces uniform creation/generation of digits.
- Unlike Denoising Autoencoders, InfoGan give freedom to specify which input componenet is to be retained.
- While providing a categorical variable and then incorporating the MI into the generator is fine, it's actually remarkable that of all the possible things to learn (height, tilt, etc), it learns the class information. 

Some conclusions from the final section:
- Tying latent codes with generator output can yield interpretable representations.
- Choice of latent codes is data dependent.
- Why it does what it does is still not clear...

---

### InfoGan paper (2016 - Chen, Duan and OpenAI folks) : https://arxiv.org/pdf/1606.03657.pdf

- Learning dis-entangled representations which represent salient attributes of data is useful for downstream tasks. 
- These dis-entangled representations might allocate separate dimensions for different facial attributes like eyes, color, hair color, etc.
- Most unsupervised research is driven by generative modelling : in trying to learn "how to create observed data", you *might* also learn good representations.
- **Previous methods all required supervision for arriving at dis-entangled representations.** *(explore this more!! - Section 2)*.


**Introducing latent codes:** It would be helpful if we can recover latent properties of MNIST digits like tilt and thickness in an unsupervised manner. 

- Rather than using a single unstructured noise vector as input, we split the noise into 2 parts : the unstructured vector $z$ and the latent code $c$ which will target salient, structured, semantic features.
- Factored distribution over latent codes simply as product of independent codes.
- Provide the generator with both the $z$ and $c$; the generator form becomes $G(z, c)$. The purpose is that $I(c; G(z, c))$ should be high.


Mutual information $I(X; Y)$ measures the amount of information learned about variable $X$ from knowledge of variable $Y$.
$$
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$
This gives an interpretable definition : the reduction of uncertainity in $X when $Y$ is observed. 

- Calculating $I(c; G(z,c))$ directly involves calculating the posterior $P(c|x)$. Instead, we introduce an auxilary function $Q$ and aim to estimate the variational lowerbound $L_I(G, Q)$.
- In practice, $Q$ is a neural network which shares all convolutional layers with $D$ except for final layer which outputs parameters for $Q(c|x)$. 
    - For categorical codes, use a softmax.
    - For continuous, use a factored gaussian.
- Convergence is faster than normal GAN.

---

# (digression) A brief history of attention ...

An attention mechanism/block takes in $n$ arguments $y_1, y_2, ...y_n$ and a context $c$. It returns a vector $z$ summarizing all inputs while giving more weight to the inputs which are more important given the context.

**BREAKDOWN**
1. The input (think of an entire image) has been broken down into smaller inputs (the $y_1, y_2, ..., y_n$). At every "step", you have a part of the data $y_i$ and the context $c$.
2. The interaction between the context $c$ and the current input $y_i$ are computed independent of other $y_{j \neq i}$. Think of this like a vanilla RNN computation with an additional constant input of $c$ for every step.
3. Once all input interactions have been independently computed, we compute the softmax over all interactions.
4. The output $z$ is a sum of all inputs weighted by their softmax scores.

The process above makes more sense in a RNN scenario (encoder-decoder type framework). However, when the order of inputs does not matter, one can consider independent states : the attention mechanism can be fully feed-forward as in [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf) and [Memory Networks](https://arxiv.org/pdf/1503.08895.pdf)

Naive attention mechanisms are prodigal. For a 50 word input to 50 word translation, we are calculating 2500 values. This seems wasteful because human attention mechanism usually focuses on a few things.