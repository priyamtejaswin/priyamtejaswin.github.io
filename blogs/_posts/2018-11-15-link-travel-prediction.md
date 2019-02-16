---
layout: post
title: "[Notes] Link Travel-Time Prediction"
excerpt: "Review of recent approaches in literature for estimating/predicting the link/trip travel time."
---

Review of recent approaches in literature for estimating/predicting the link/trip travel time.

## Core References
1. Link Travel Time Prediction from Large Scale Endpoint Data (SIGSPATIAL 2017) - <https://dl.acm.org/citation.cfm?id=3140006>
2. Probabilistic estimation of link travel times in dynamic road networks (SIGSPATIAL 2015) - <https://infolab.usc.edu/DocsDemos/mohammad_1_2015.pdf>
3. Non-Parametric Estimation of Route Travel Time Distributions from Low-Frequency Floating Car Data (Transportation Research Part C: Emerging Technologies, 2015) - <https://people.kth.se/~jenelius/RJK_2014.pdf>
4. T-Drive: Driving Directions Based on Taxi Trajectories (SIGSPATIAL 2016) : <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/T-Drive-20Driving20Directions20Based20on20Taxi20Traces.pdf>
5. A simple baseline for travel time estimation using large-scale trip data (SIGSPATIAL 2016) : <https://dl.acm.org/citation.cfm?id=2996943> (**in progress** - was ignored by other papers for some reason...)
6. When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks (AAAI 2018) : <https://www.microsoft.com/en-us/research/uploads/prod/2018/01/travel-time-estimation-dnn.pdf>
7. Arriving on time: estimating travel time distributions on large-scale road networks (2013) : <http://jackdreilly.github.io/papers/bestroute.pdf>

---

## 1. Link Travel Time Prediction from Large Scale Endpoint Data (SIGSPATIAL 2017) - <https://dl.acm.org/citation.cfm?id=3140006>
- Tries to estimate the link times when link times are not given; only the TOTAL TRIP TIME is given.
- Uses NYC Cab Data.
- Procedure:
    - Start with only trips: O coord, D coord and duration.
    - Eliminate *ambiguous* trips (i.e. trips with more than 1 possible path, since multiple paths are possible).
    - The average link time $x_k$ is the expected value of the link travel times.
    - The average trip time $T_p$ is the expected value of all the trip times.
    - Minimize $AX - T$ where $A$ is (\|$links$\|x\|$trips$\|)coefficient matrix: value of $A_kp$ is 1 if link $k$ exists in trip $p$, else 0.
    - Add non-negative constraint (nnls) for ensuring that estimated link times are >= 0. **THIS WAS AN ISSUE WITH THE BASELINE MODEL PPE.**
    - Only baseline is a PPE model which correlates the trip times with the fare data and tries to estimate the link time.
    - Other models were skipped - paper claimed they were "infeasible" and "would not scale to large networks".

- All results presented for ($DayType$ x $DayHour$) cuts; why not weekly cuts as well?
- Some relevant papers were skipped.
- If final task is to predict TripTimes, then no reason for why a discriminative model could not be used.

We already have the "link" data for different trips - initial approach could just be averaging these link times for different cuts ($Month$ x $Day$ x $DepartureHour$).

My hunch is that the hop travel times for these cuts should be consistent; trip travel time variations in NYC were in order of a 2-3 minutes. *We* will be averaging travel times on highways. I suspect the variability in the SLA is coming from the assests and not the connections.

## 2. Probabilistic estimation of link travel times in dynamic road networks (SIGSPATIAL 2015) - <https://infolab.usc.edu/DocsDemos/mohammad_1_2015.pdf>

- Core task is to predict the travel time over a route *probabilistically*. Innovation is to estimate link travel times and then combine these for a route.
- Travel times conditioned only on the link enterance time. 
- Paper introduces statistical models for estimating *probabilistic link travel times* or *pltt*. Also introduces methods to probabilistically estimate the route travel times. With this, it also proposes a metric for comparing the "reliability" of two routes.
- Dataset is private.
- Procedure:
    - Unlike **Link1**, historical link data is available.
        - Define $t_s$ as the *StartTravelTime* at which we want to predict the travel time.
        - Define $t_q$ as the *QueryTime* at which the query at $t_s$ is made.
        - *pltt* between a link pair *(i, j)* is defined as $c^{t_s}_{ij}(x)$ : **probability of taking $x$ seconds to traverse link *(i, j)* starting at time $t_s$**.
    - Historical link travel data can be fit to continuous or discrete random variable.
        - For continuous, fit to normal distribution.
        - For discrete, quantize the time and get the PDFs.

    - This can be used directly, or it can be more "real-time" for queries in the *very* near future. 
        - (1) $\theta.Hist + (1 - \theta)Curr$
        - (2) Through similar historic data based similarity.
    - Rest is on estimating route-reliability and evaluating the "match" performance of the routes.

Again, provided we frame the problem in a similar manner - predict travel times given the enterance times - I belive we should be good.


## 3. Non-Parametric Estimation of Route Travel Time Distributions from Low-Frequency Floating Car Data (Transportation Research Part C: Emerging Technologies, 2015) - <https://people.kth.se/~jenelius/RJK_2014.pdf>

- Core task is to *route travel time estimation*.
- Uses FCD - low frequency GPS car data.
- This paper argues *against* the Niloy paper, stating that "**drawback of the link-based approach is that statistics of the route travel time distribution (apart from the mean value) are not straightforward to derive from the travel time distributions of the constituent links.**"
- It cites the approach taken by Westgate (2013) : "**avoids the problems of aggregating link travel times into route travel times, but does not utilize the information from intermediate FCD reports.**"
- No assumptions on the functional form of the data.
- Accounts for travel time variability due to (departure)time of day.
- Focus on utilizing partially overlapping routes.
- Paper discusses many of the biases which are often unaccounted for while estimating travel times using FCD:
    - partial overlaps
    - oversampling of sections of a route
    - which vehicle fleet is used
    - are the cars travelling in bus lanes?
- System evaluated on Stockholm's taxi-fleet GPS data; pings recorded every 2 minutes.
- The 'procedure' is divided into 3 main steps:
    - transformation
    - weighting
    - aggregation

- **Transformation**
    - Transform the raw GPS data into an *instance* of route travel. Once the data is mapped onto a physical road network, you group/concat observations of the same trip into a *trace*.
    - ![](https://i.imgur.com/rFXvFiZ.png)
    - The figure above shows multiple *traces*: 1-3, 1-4, 2-3 and 2-4. The paper proposes to use the trace which has the highest weight as defined by a kernel function.
    - Other steps in **Transformation** are to account for GPS observations which partially cover adjacent routes (*allocation*); or when the start-time is not known (*route entry extrapolation*); 
    - The final output of this step is some scaled travel-time for the selected *trace*. This is ONE single instance of a route travel-time observation, $T_i$.

- **Weighting** We weight every travel-time observation $T_i$ to determine it's influence on the final travel-time statistic.
    
- **Aggregation** Final step is to compute relevant statistics for the route given the different observations ($T_i$) and their weights ($w_i$). Formulas in the paper.

Not quite sure what to make of this approach. Final trip time is once again, a weighted mean. Some clever ideas to weight different trip times and also remove certain observational biases. Once again, this paper also stresses on the importace of the departure-time. While the bulk of the data is the consignment/gate time, TMS has started to record GPS data from trucks... could be useful there, but we'll have to solve some very basic problems like map-matching before that. *Also applies to the FE gps data from the OLD app which is still being captured but not used.*

## 4. T-Drive: Driving Directions Based on Taxi Trajectories (SIGSPATIAL 2016) : <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/T-Drive-20Driving20Directions20Based20on20Taxi20Traces.pdf>

*Alt Title:* **How to incorporate driver intelligence for suggesting fastest routes b/w 2 points.**

The notion of landmarks makes the system more "intuitive" ... as opposed to using the exact lanes or turns??? The assumption is that a seasoned driver will always takes the shortest routes b/w two points. Additionally, said driver will be aware of the time-dependent road issues - traffic, accidents, condition, etc. Thus, optimal routes *can* be inferred from the driver trajectories.

Two most relevant results from the paper:
1. This paper (like all others above) also conditions the data on weekday/weekend and assumes that the only other discriminating variable is the start-time. However, instead of choosing arbitrary start times, it employs a Variance-Entropy clustering algo. 
2. The "in-field" evaluation strategies are gold! They compared they route suggestions against Google Maps' (back in '09 when Maps used speed-constrained shortest path algotithms) and showed that for 75% of the time, their route was 14% faster than Google's.

Notes

- Paper uses a dynamic, time-dependent (and weekday/weekend dependent) landmark graph. Landmark == road segment.
- Distribution of travel time b/w 2 landmarks in different time slots is arrived by a VE-clustering algorithm.
- Naturally incorporates a lot of things: wait times, signals, bad roads, etc.
- Major modelling challenges:
    - Coverage - user queries can start/end anywhere; trajectories will be limited. In sufficient number of taxies per segment.
    - Sampling - results will vary by frequency; likelihood of traversing alternate routes also increases with decrease in sample rate.

1. Build a time-dependent landmark-graph: 
    - select top-k road segments
    - edge exists if trajectory count support exists
    - compute travel-time distribution using VE-clustering - this gives a **time-dependent landmark graph**
    - separate graphs for Weekday and Weekend.
    - Section 4.2 (with figure) for graph construction.
    - **VE-clustering to automatically split the time.**

<div class="post-image">
<img src="/assets/images/link-prediction-tdrive.png">
<p><em><font size="-1">Graph consitruction for T-Drive.</font></em></p>
</div>

2. VE-clustering:
    - sort all travel times for a edge
    - keep partitioning on the **TravelTime** in a binary-recursive way; criteria is to minimize average variance post split
    - this will give a set of **TravelTime** clusters
    - NOW, cluster the departure times

3. Compute a route:
    - query$(q_s, q_d, t_d)$ : StarPoint, DestinationPoint, DepartureTime.
    - **Time Dependent Fastest Path Problem** -- solved.

- EVALUATION (the good stuff is in Section 6.4)
    - Beijing. 105k nodes, 141k road segments; 33,000 taxi trajectories over 3 months.
    - In Field 1 - same driver traverses routes at different times. (360 paths over 10 days)
    - In Field 2 - two drivers follow different routes to the same location AT THE SAME TIME. (60 paths over 6 days)
    - Measure R1 is % of times TDrive was better than baseline (Google). Measure R2 is extent to which it is better. 