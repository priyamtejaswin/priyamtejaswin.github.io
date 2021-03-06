---
layout: post
title: "NBA-goat (part 2)"
excerpt: "Discovering the <strong>g</strong>reatest NBA team <strong>o</strong>f <strong>a</strong>ll <strong>t</strong>ime using statistics. This is part 2 which covers ADF."
image: "/assets/images/nba-goat-adf.gif"
---

This is a continuation of the NBA-goat project where I explore different statistical methods for estimating the *true*-skill of a team over time. Part 1 covered ELO. Part 2 covers ADF (Assumed Density Filtering).

### A note about the notebooks.
I wrote this tutorial before I was aware of hackmd.io. At the time, I would write massive Jupiter notebooks with all the  math, code and figures. The **nbviewer** link will take you to the rendered notebook - this is the recommended way of viewing the project. The 2nd cell of the notebooks contains some javascript which hides all the input code cells for a pleasant reading experience. If you're interested in the code, then please click on the **here** link in the 2nd cell.

### Part 2. ADF:
[https://nbviewer.jupyter.org/github/priyamtejaswin/nba-goat/blob/master/nb-adf_team.ipynb](https://nbviewer.jupyter.org/github/priyamtejaswin/nba-goat/blob/master/nb-adf_team.ipynb)

- [x] Start by explaining the 2 core operations (convolution, greater-than).
- [x] Explain the clutter problem and the complexity involved with calculating the exact posterior.
- [x] Derive the parameter updates for the clutter problem using ADF.
- [x] Visualise the update procedure.
- [ ] Setup the skill estimation problem in context of ADF.
- [ ] Derive updates using ADF.
- [ ] Apply on NBA data.
- [ ] Compare against mov-Elo from previous notebook.

Scroll down to the last cell to view ADF in action while estimating the true mean in noise!