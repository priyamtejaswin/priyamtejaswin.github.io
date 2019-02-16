---
layout: post
title: "[Notes] Adversarial Examples"
excerpt: "A BFS into adversarial example generation."
---

I and Akshay Chawla had some ideas on generating adversarial examples for audio/speech processing systems. We noticed that none of the existing attacks worked "in the wild" and saw an opportunity to propose something new, possibly inspired from the vision community.

This post will review papers which explore "adversarial example" generation for vision as well as audio:
1. Intriguing properties of neural networks - Szegedy, Goodfellow (2014) . One of the first papers to explore adversarial examples in the functional spaces of DNNs. <https://arxiv.org/pdf/1312.6199.pdf>
2. Explaining and harnessing adverarial examples - Goodfellow and Szegedy (2015) . A more formal approach and analysis. Introduces the FGSM. <https://arxiv.org/pdf/1412.6572.pdf>
3. Adversarial examples in the physical world - Kurakin and Goodfellow (2016) . <https://arxiv.org/pdf/1607.02533.pdf>


# Intriguing properties of neural networks - Szegedy, Goodfellow (2014)

### 1. Introduction
Discusses two counter intuitive properties:
1. **Semantic meaning of individual units.** The contention is that it's the entire space of activations - rather than the individual units - which contain the semantic information. This is echoed by Mikolov for continuous representations of word-vectors: it is the combined directions of all units and not that of a single unit that lead to semantic interpretibility. 
2. **The input-output mappings learnt by the network are fairly discontinuous.** By applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction. These perturbations are found by optimizing the input to maximize the prediction error and are appropriately referred to as “adversarial examples”.

These "adversarial examples" are robust - they are valid across architectures, hyper-parameters and training sets.

### 2. Notation
They denote $x \in \mathbf R^m$ as the input image and $\phi(x)$ as the activations of some layer. 

### 3. Units of: $\phi(x)$
To analyse a neural network, you look at the activations of a hidden unit as a meaninigful feature. You hunt for input images which maximize the activation of this single feature(i.e. unit).

"Basis" of some subspace V is that set of vectors which are independent and span V. If these are orthonormal, then it is called the "standard basis". 

Turns out, you will find semantically related images for any random basis. And you will not always find semantically similar images. 

### 4. Blind spots
Main result is that the for deep neural networks, the smoothness assumption that underlies many of the kernel methods does not hold true. 

Some optimisation stuff in the approach - i'll come back to this later if I have to.


---


# Explaining and harnessing adverarial examples - Goodfellow and Szegedy (2015)
They argue that the primary cause for vulnerability to adversarial perturbations is the linearity of the networks. This "linear" views enables a fast method for generating adversarial examples. 

Let $\theta$ be the parameters of the model, $x$ be the input to the model, $y$ be the targets for $x$ and $J(\theta, x, y)$ be the cost used to train the network. 

With this, the optimal max-norm constrained perturbation of $J$ is
$$
\eta = \epsilon\ \mathbf{sign}(\nabla_x\ J(\theta, x, y))
$$

Adding $\eta$ to the original samples leads to 99% error rate on MNIST with average 80% confidence.

In summary:
- AdvExamples exist because of models being too linear instead of non-linear.
- Direction of perturbation matters most.
- The direction should carry over to different clean examples.
- Adversarial training helps.
- Linear models cannot resist adversarial perturbations.
- Ensembles are not resistant to AdvExampes.

# Links to Speech Recognition 

***Notes - https://hackmd.io/s/HyM-o8STM***

1. DeepSpeech overview - https://www.youtube.com/watch?v=9dXiAecyJrY&feature=youtu.be&t=13874 
2. http://lxmls.it.pt/2017/talk.pdf ??
3. Baseline Implementation of (1): https://github.com/baidu-research/ba-dls-deepspeech , this is a baseline implementation of Baidu deepspeach using an older version of Keras + theano. 
4. Summary of adversarial procedures(including 1-pixel and DeepFool) - https://arxiv.org/pdf/1801.00553.pdf   https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
5. Carlini's paper - https://arxiv.org/pdf/1801.01944.pdf
6. Carlini's paper page - https://nicholas.carlini.com/code/audio_adversarial_examples/
7. Original DeepSpeech paper - https://arxiv.org/pdf/1412.5567.pdf
8. Raw Waveform for audio classification - https://arxiv.org/pdf/1712.00866.pdf


Next steps:
- Establish the difference between MFCC and spectograms. Check if latter can ALSO be fooled adversarially.
- Check if raw-waveform classifiers can be fooled adversarially.
- Try the new methods discussed in (4) which were not considered by Carlini for evaluation. Especially forcus on decreasing the amount of noise/distortion added by something like one-pixel attacks (perhaps something like "single note" attacks?)
- Understand/proove why the existing technique fails at physical attacks. Try to get a physical attack working.

---

# Literature Review

### Audio Adversarial Examples: Targeted Attacks on Speech-to-Text (https://arxiv.org/pdf/1801.01944.pdf - 2018)
- Some work done for discrete signals (like text/nlp adversarial for reading comprehension https://arxiv.org/pdf/1707.07328.pdf). It's primitive yet effective.
- Constructing targeted audio attacks is tough.
- Carlini (USENIX 2016 https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_carlini.pdf) creates stealth audio but it's just heavily distorted sound.
- Deep Learning and Music Adversaries (2015 - https://arxiv.org/pdf/1507.04761.pdf) tries it for music category classification using CNNs - not sure why Carlini's paper puts this under untargeted attacks???
- other SOTA is Houdini (https://arxiv.org/pdf/1707.05373.pdf - 2017). Fools any gradient-based learning machine by generating adversarial examples directly tailored for the task loss of interest be it combinatorial or non-differentiable. They try ASR fooling on DeepSpeech2, but it works on phonetically similar samples.
- **Problem Identified:** targeted attacks in audio domain seem to be much more difficult.
- Their contribution is targetted attacks on DeepSpeech. Given a natural waveform $x$, they are able to construct a perturbation $\delta$ - that is nearly inaudible - for which $x + \delta$ is recognized as **any desired phrase**. They use strong iterative optimisation. 
- Much like our idea, this seems to be a fixed/static mapping rather than a "one method fits all approach". Their threat-model is deemed successful if the output matches the exact "targetted" phrase.
- Current work assumes no extra noise ; i.e. it's NOT a physical attack.
- Initial formulation was a constrained minimisation problem where they DID consider the range of $\delta$ , but then they resorted to Szegedy et al's 2014 paper (Intruiging properties of neural netowrks.) New formulation is :
**minimize $dB_x(\delta) + c\ .\ l(x + \delta, t)$**
Here $dB_x$ is the decible value, $l(x', t)$ measures the loss between the purturbed sample $x + \delta$ and the target phrase $t$. $l(x', t)$ should be less that equal to $0$ or equivalent to $C(x') == t$.
- Finding a function for $l()$ is non-trivial since it involves complex decoding (side: there are seq-to-seq papers which optimise for beam-search directly).
- Actual approach is by optimising for $dB_x(\delta) <= \tau$, solving for a while and then reducing $\tau$.
- Their approach takes 1 hour/example on GPU. 
- Another **major** contribution is an improved loss function. This was due to the decoding process in audio phrase generation as opposed to getting the probability estimate in image classification - THIS GIVES ME HOPE FOR THE HOTWORD TASK!
- There are seq-to-seq optimisation models which directly work on beam-search. (https://arxiv.org/pdf/1606.02960.pdf - 2016). Perhaps we can find something useful here?
- They show that iterative optimisation is better than FGSM - DeepNeuroEvolution should work ; reasons cited for poor FGSM performance are MFCC and LSTM. CNN architecture should work better then?

Biggest opportunities:
1. Physical attacks.
2. Making a "semi-"universal perturbation; perhaps focus on something that attacks something that's common in all speech (all vowels or major cononants)?
3. Check for transferrability across tasks and architectures?


### Hidden Voice Commands (https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_carlini.pdf - 2016)
1. Black box attack: Not Useful since it requires a HIL component.  
2. White box attack: Underlying system is CMU Sphinx (voice->MFCC->13-dim vector->1st+2nd derivatives->39-dim vector->GMM->HMM->phenomes to probability over words)
Simple attack: find a target MFCC (y) , objective --> f(x) = (MFCC(x) - y)^2. Results not better than black box attack. 
Improved attack: "First,rather than targeting a specific sequence of MFCCvectors, we start with the target phrase we wish to produce, derive a sequence of phonemes and thus a sequence of HMM states, and attempt to find an input that matches that sequence of HMM states. This provides more freedom by allowing the attack to create an input that yields the same sequence of phonemes but generates a different sequence of MFCC vectors."
A lot of hand-tuning was used in this code. 


### Crafting Adversarial Examples For Computational Paralinguistic Applications (OHYEAH) (https://arxiv.org/pdf/1711.03280.pdf - November 2017)
- look at how they present the results/differences in the audio wavforms.
- they chose not to show decible noise difference even though Carlini's work was before their's???

end-to-end method to generate adversarial examples by directly perturbing the raw audio waveform rather than specific acoustic features. 
Computational linguistics = What the person is saying (spech to text)
Computational para-linguistics = How the person is speaking (speech emotion recognition, speaker verification). (THIS PAPER)

References Carlini 2016.
Kereliuk and Larset (look below): Perturbation on Spectograms, difficult to get back the time domain signal due to overlapping windows. 
Iter, Huang, and Jermann 2017: extract MFCC features and perturb it. Difficult to get back the original sound signal due to MFCC being a lossy transform. 
Both the above methods apply the perturbation on extracted features, thus involving another step where the perturbed feature has to be converted back to the sound domain, this causes loss of fidelity and the perturbed sound signal actually sounds different from the original signal.

Contributions: (1) perturbation on raw audio waveform rather than extracted features. (2) discover vanishing gradient problem in RNN so substitute RNN with a CNN based NN ?? (3) Experiments on 3 different para-linguist tasks. 

Minimize ‖η‖ s.t. f(x+η) /= f(x)
η is noise vector 
x is original sample 
f(..) is the NN. 
Attack method: FGSM 
Model: WaveRNN and WaveCNN 

Goes on to describe vanishing gradient problem in RNN, i.e gradient of loss w.r.t input is very small. 

Experiemtns on 3 para-linguistic tasks : Gender recognition(binary), Emotion recognition(binary), Speaker recognition(4-class). 

Interesting notes on the perturbation: Perturbation on raw waveform is small er in magnitude than perturbation on featurebased methods. The perturbation on raw waveform (henceforth called pertraw) sounds like "normal" noise. Humans have no problem performing the 3 para-linguistic tasks with pertraw added to the original signal, but it wreaks havoc on the the deep nn. If we compare spectograms of the original and adversarial example, pertraw covers all of the available spectrum, meaning it might be immune to simple filtering based defences. 


### Did you hear that? Adversarial Examples Against Automatic Speech Recognition (OHYEAH)(https://arxiv.org/pdf/1801.00554.pdf - 2018)
**Problem to attack**: Tensorflow Dataset of Keyword spotting 
**Base model**: Based on model by "Convolutional neural networks for small-footprint keyword spotting" which is the model used in the tutorial. It is MFCC based. 
**Attack method**: Non-gradient based Genetic ALgorithm, basically, keep mutating the x (input) (original benign audio clip), till you hit f(x_adv) == new target. 

- they use GA for attacking the least significant 8 bits
- make no use of the gradient information (perhaps neuroevolution might be better)
- the system appears to be biassed since it's not trained to detect noise
- "recruited" 23 "participants" to evaluate quality of attacked samples.

### Generating Adversarial Examples for Speech Recognition (http://web.stanford.edu/class/cs224s/reports/Dan_Iter.pdf - 2017)
Target - automatic speech recognition systems (ASR)
Model - WaveNet (pre-trained) , inputs are MFCC features. 
create features in MFCC domain, invert them back to sound domain. 
Method: FGSM and Fooling gradient sign method. On both single work outputs and sentences. 

### An Overview of Vulnerabilities of Voice Controlled Systems (https://arxiv.org/pdf/1803.09156.pdf - 24th March 2018)
There is a section on ML adversarial method based attacks that mentions 3 papers: 
1. Carlini and D. Wagner, “Audio adversarial examples: Targeted attacks on speech-to-text --> Deepspeech targeted attack. 
2. Paralinguistics paper --> adversarial examples can misclassify gender and  identity of the speaker. 
3. Houdini Paper --> Show that adversarial attacks are transferable to unknown and different ASR models. 
Only the first paper mentions that over-the-air attack is not possible, 2nd and 3rd paper says nothing aboutover-the-air attack performance. 

### Deep Learning and Music Adversaries ; music content analysis (https://arxiv.org/pdf/1507.04761.pdf - 2015)
- Main experiments are over music genre identification.
- CDNN model with softmax output and spectogram features (STFT). Initial convolutional windows are of (400 x 4). Reason for long rectangular windows is to capture strong harmonic structures which span the audible spectrum. 
- Modification of the Sedgezy's line-search approach; cause for failure of original was inability to correctly back-map to time domain. They use the Griffin-Lim algorithm to project the adversarial example back to time domain. An updated and "available" version of the algorimth is here - https://perraudin.info/publications/perraudin-note-002.pdf

### Adversarial Diversity and Hard Positive Generation ; a new way to generate diverse adversarial samples(https://arxiv.org/pdf/1605.01775.pdf - 2016)
Main contributions:
- Introduced PASS to quanify adversarial images.
- New approach to generate large number of adversarial images.

[SIDE]Interesting approach by Sabour 2016 where they use a guide image to perturb the internal representation to get closer to the guide's representations.

- Related work covers Sedgezy's 2015 paper which uses box-constrained LBFGS to find the smallest perturbation in the input space that causes the perturbed image to be mis-classified as the chosen target label.
- Related work also covers Goodfellow's FGSM for creating small perturbations. Stepping into the direction of the sign of the gradient of loss w.r.t. input image continuously reduces the classification score of the original. Another work by Sabour et al. (2016)

Contributions: FGV and Hot/Cold
- $\theta$ : parameters
- $x \in [0,255]^m$ : $m$-pixel input image
- $y$ : label of $x$
- $J(\theta, x, y)$ : cost
- $f$ : learnt classifier
- $n$ : total classes(labels)
- $B_l(.)$ : backprop operator which extends all the way to the input

Our goal is to produce perturbation $\eta$ such that $\tilde x = x + \eta$ is misclassified , i.e. $f(\tilde x) \neq y$. To generate hard-positives, we scale $\eta$ by constant >= 1.

**FGV** extends FGS by considering a scaled version of FGS instead of just moving along the sign. The intuition here is to decrease the response to the class of interest by following the direction of the gradient of loss. This leads to purtubed examples with less structural damage than FGS. Can we have more ?

**Hot/Cold** aims to consider derivatives w.r.t. other layers in order to augment the overfit decision boundaries created by naturally moving towards a specific targetted class.

### HOLY GRAIL - Raw Waveform-based Audio Classification Using Sample-level CNN Architectures (https://arxiv.org/pdf/1712.00866.pdf - 2017)

- Lower layers while using raw-waveforms should be able to identify all possible phase-variations of periodic waveforms. Think of a sine wave shifted *right* along time; this will probably make no difference to a mel-spectogram but can cause issues with time-domain analysis. Frequency domain techniques are invariant to phase variations within a frame since they only look at the frequency magnitude.
- This problem is analogus to "translational invariance" in image domain. For sound detection and auto-tagging, authors used VGG-style 1D CNN models. These turned out to be very effective. 
- This was further enhanced using ResNet style connections. 
- Model architectures in https://arxiv.org/pdf/1710.10451.pdf and https://arxiv.org/pdf/1703.01789.pdf

reference 11[FROM THE FEB 2018 arxiv PREPRINT]
- How multi-level aggregation affects performance for music tagging?