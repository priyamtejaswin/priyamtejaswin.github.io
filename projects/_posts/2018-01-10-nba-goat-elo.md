---
layout: post
title: "NBA-goat (part 1)"
excerpt: "Discovering the <strong>g</strong>reatest NBA team <strong>o</strong>f <strong>a</strong>ll <strong>t</strong>ime using statistics. This is part 1 which covers ELO ratings."
image: "/assets/images/nba-goat-elo.png"
---

To find the **greatest-of-all-time** using statistics. I started this project to explore different methods for estimating relative skill of NBA teams. My inspiration was [FiveThirtyEight.com's CARMElo system](https://projects.fivethirtyeight.com/2018-nba-predictions/). The aim is to dive-deep into every method, derive the update equations and discuss the pros and cons. I intend to cover the following methods:
1. Elo and common extensions.[DONE]
2. Assumed Density Filtering.[IN PROGRESS]
3. Expectation Propagation(given that this NBA data has already ocurred and can be used for batched inference.)
4. Extensions to EP(score difference).

The complexity of methods increases according to the [standard, universally accepted W3D difficulty rating system](http://agentpalmer.com/wp-content/uploads/2014/10/Setting-your-Wolfenstein-3D-Difficulty-Level.jpg). You have been warned.

### A note about the notebooks.

I wrote this tutorial before I was aware of hackmd.io. At the time, I would write massive Jupiter notebooks with all the  math, code and figures. The **nbviewer** link will take you to the rendered notebook - this is the recommended way of viewing the project. The 2nd cell of the notebooks contains some javascript which hides all the input code cells for a pleasant reading experience. If you're interested in the code, then please click on the **here** link in the 2nd cell.

---

### Part 1. Elo:
[https://nbviewer.jupyter.org/github/priyamtejaswin/nba-goat/blob/master/nb-elo_vanilla.ipynb](https://nbviewer.jupyter.org/github/priyamtejaswin/nba-goat/blob/master/nb-elo_vanilla.ipynb)

- [x] Tracking NBA franchises through changes in names and cities.
- [x] Explain Elo with its core assumptions and apply the vanilla Elo on nba data.
- [x] Extend the base model to account for score difference(mov-Elo).
- [x] Extend the base model to account for home-court advantage(hca-Elo).
- [x] Finish with interactive visualisation of Warriors and Bulls.
- [x] Discuss and segue to ADF and TrueSkill.

Scroll down to the last cell for an interactive visualisation for two of my favorite teams!