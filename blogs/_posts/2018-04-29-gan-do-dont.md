---
layout: post
title: "[Notes] Understanding Generative Adversarial Networks"
excerpt: "The GAN model deep-dive with do's and don'ts."
mathjax: true
---

How could one possibly ignore all the hype? This is being used in everything from images to audio and even [e-commerce](https://arxiv.org/pdf/1801.03244.pdf)! These are my notes from [original 2014 GAN paper](https://arxiv.org/pdf/1406.2661.pdf) and the excellent [NIPS tutorial in 2016](https://arxiv.org/pdf/1701.00160.pdf).

*The headings have section numbers which correspond to the NIPS Tutorial document.*

<div class="post-image">
    <img src="/assets/images/gan-model.png">
    <p><em><font size="-1">GAN block diagram.</font></em></p>
</div>

## 3.1 The GAN framework
- Setup game between 2 playes : generator and discriminatior.
- Generator creates samples which should belong to the same distribution as the training data. Discriminator is supposed to distinguish the real from the fake.
- Discriminator learns from supervised learning(using labelled fake VS real samples); the generator is trained on fooling the discriminator.
- Discriminator is defined by function $D$ that takes $x$ as input and uses $\theta^{(D)}$ as parameters.
- Generator is defined as function $G$ that takes $z$ as input adn uses $\theta^{(G)}$ as parameters.
- Cost functions for both G & D are defined in terms of both of their parameters combined:
-- Discriminator must minimize $J^{(D)}(\theta^{(D)}, \theta^{(G)})$ while only controlling for $\theta^{(D)}$.
-- Generator must minimize $J^{(G)}(\theta^{(D)}, \theta^{(G)})$ while only controlling for $\theta^{(G)}$.
-- *The definition of a games makes sense since each player can only optimise his parameters and not ther other player's.*
-- *Solution to an optimisation problem will be the global minima. Solution to a game will be Nash equilibrium.*
-- In this case, the Nash equilibrium is a parameter setting $(\theta^{(D)}, \theta^{(G)})$ that is local minima of $J^{(D)}$ w.r.t. $\theta^{(D)}$ and is the local minima of $J^{(G)}$ w.r.t. $\theta^{(G)}$.

**Note on Generator design**
- $G$ can be any differentiable function. When $z$ is sampled from some prior distribution, $G(z)$ yields a sample of $x$ from $p_{model}$.
- Usually, a DNN is used to represent $G$. But, the inputs to $G$ do not need to be the inputs of the DNN; people have employed other strategies as well.
- If $p_{model}$ should have full support of $x$ space, then the dimensions of $z$ should be at least as large as dimensions of $x$.

**Note on training process**
Entire model trained end-to-end by simultaneous SGD.

1. Two mini-batches are sampled: a mini-batch of $x$ from the dataset and a mini-batch of $z$ from the prior. 
2. Gradient updates w.r.t. to $\theta^{(D)}$  and $\theta^{(D)}$ are made simulataneously. 
3. Varying opinions on training the two players for different number of steps before the update. Goodfellow believes single-step simultaneous updates are best.
4. Adam is your friend.

## 3.2.1 Discriminator cost function

Different GAN models (or games) use the same discriminator cost $J^{(D)}$. 

$$
J^{(D)}(\theta^{D}, \theta^{G}) = -\frac{1}{2} \mathbf{E}_{x \sim p_{data}} log\ D(x) - \frac{1}{2} \mathbf{E}_{z \sim p_{gener}} log\ (1 - D(G(z)))
$$

This expression is an extension of the binary cross-entropy loss function.
1. There are two terms. One for $x$ sampled from data and one for samples from the generator, i.e., $G(z)$.
2. Following from (1.) the "truth" values for the first term will be 1 and for the second term will be 0. The discriminator needs to correctly classify the real samples as well as the adversarial samples.
3. The notion of the $\frac{1}{2}$ times expectation comes from the fact that we are summing over exactly half of the samples from each source.
4. The 2nd term in the equation can be re-written in terms of the generator $p_{generator}$ by replacing $G(z)  = x$ where $x \sim p_{generator}$.

The optimal value for the discriminator $D$ given a fixed $G$ is $\frac{p_{data}}{p_{data}+p_{gener}}$ . The paper refers to $p_{gener}$ as $p_{model}$.


## 3.2.2 Generator cost function(s)
### Minimax game

For a zero-sum setting, the generator cost will just be the negative of the discriminator cost. And since the generator function $G$ is specified in the discriminator cost, that alone can be used as the complete value function for the minimax game: $\theta^G$ is minimized and $\theta^D$ is maximized.

### Heuristic, non-saturating game

In a minimax setting, the discriminator maximizes the cross-entropy and the generator minimizes the same cross-entropy. If the discriminator dimisses a sample as fake with very high probability, the gradient for the generator will be diminished.

To overcome this, the generator can maximize the discriminator's probability of being incorrect.
$$
J^{(G)} = -\frac{1}{2}\mathbf{E}_z\ log\ D(G(z))
$$

## 3.2.5 Choice of divergence important?

KL(data \|\| model) vs KL(model \|\| data) ?? ***refer the tutorial section 3.2.5 again later.***

## 3.3 The DCGAN

<div class="post-image">
    <img src="/assets/images/dcgan-diag.png">
    <p><em><font size="-1">Deep Convolutional Generative Adversarial Network diagram.</font></em></p>
</div>

- Batch-norm for D and G seprately. 
- Last layer of G and first layer of D are not batch-normed so that the model can learn the correct mean and scale of the data.
- No pooling or un-pooling. For increasing spatial dimensions, increase the convolutional strides.
- Use Adam instead of SGD with momentum.

## 4. Le hacks...

*Soumith's repo https://github.com/soumith/ganhacks#authors*

1. Use labels if available (also called class-conditioning): sample quality improves greatly. Just using the labels with the discriminator is sufficient. Check Denton et.al (2015) and Salimans et.al (2016).
2. One-sided label smoothening: to encourage the discriminator for returning soft-probabilities rather than extremely confident classification. Check Salimans et.al (2016). In tensorflow, while computing the cross-entropy, simple replace `1`'s with `0.9`'s like so:
```
tf.nn.sigmoid_cross_entropy_with_logits(d_on_data, 0.9)
```
This prevents extreme interpolation in the discriminator; it will be penalized for predicting extremely large logits.

DO NOT smoothen labels for the fake samples. Increasing the fake sample target value from `0` to some small constant `beta = 0.1` will cause the optimal discriminator value to behave unexpectedly. 

For $\alpha$ decrease in the real targets and $\beta$ increase in the fake targets, the optimal value for $D$ changes to
$$
\frac{(1 - \alpha)\ p_{real}(x) + \beta\ p_{gener}(x)}{p_{real}(x) + p_{gener}(x)}
$$
If $\beta$ is `0`, it becomes the real-smoothening case. If non-zero and $p_gener$ is significant, then the discriminator will start peaking near the spurious modes of p_{gener}, thereby re-inforcing incorrect behaviour of the generator.  

3. Batch-Norm: may or may not work. If batch-size is too small, then there might be huge fluctuations in the summary statistics across batches. And these might start controlling the image generation more than the input $z$.  Refer `Virtual batch-norm` in Salimans et.al (2016).
4. Balancing G and D by limiting the training steps of either: is not a good idea. There's no principled way to counter (or understand) if (or when) the discriminator is "over-powering" the generator. THEY DO NOT NEED TO BE ON THE SAME "POWER"!!!. 

GANs work by estimating the ratio of read-data density to generated-data density. This ratio is correctly estimated only when the discriminator is optimal. So it is fine even if it's overpowering the generator.

## 5. Research frontiers

### 5.1 Non-convergence
Finding the lowest point for a cost function is quite different from finding the equilibrium of a minimax game. An update for one-player might cause the other player to go up instead of down. 

**Currently, there is neither a theoretical argument that GAN games should converge when the updates are made to parameters of deep neural networks, nor a theoretical argument that the games should not converge.**

### 5.1.1 Mode collapse
Too many inputs in the $z$ space get mapped to the same output point. This may happen when the maximin solution of the game is different from the minimax solution
- maximin: maximizes the minimum loss? This is the most pessimistic strategy; it will always tell you NOT to gamble.
- minimax: minimizes the maximum loss? This is the best decision that you can make, assuming that the opponent is playing optimally.

This can cause the network to repeatedly give one type(s) of outputs. Refer to Unrolled GANs by Metz et.al (2016) and minibatch features by Salimans et. al (2016).

### 5.2 Model evaluation
This remains an open problem. Check Wu et al. (2016) where they use sampling techniques to guess the closeness to true distribution???

### 5.3 Generating discrete outputs from generator.
Nothing much.

### 5.4 How do you use the learned $z$ code?
Nothing much.

That's all folks!