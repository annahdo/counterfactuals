# HelixFlow

In this notebook we start from a data distribution that lies on a helix embedded in three dimensional space. We train a simple normalizing flow to approximate the data distribution (pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1PKfKie1KHKRHuZhjSwwiVsCYMitZFvjV?usp=sharing)).

<img src="plots/learned_distribution.png" width="800" />

We then define a classifier that divides the data points in two classes.

<img src="plots/predictions_from_classifier.png" width="500" />

Using the classifier $f$ we can generate adversarial examples by doing gradient ascent in the data space $X$. We usually walk off data manifold when we simply follow the gradient $\frac{\partial f}{\partial x}$. Alternatively we can do gradient ascent in the base space $Z$ of the flow $g$. With that we are following the gradient $\frac{\partial(f\circ g)}{\partial z}$. When we project back into the data space $x'_z = g(z')$ we can see that we stay (approximately) on the data manifold.

<img src="plots/adv_attack.png" width="500" />


We verify this by calculating the distances to the datamanifold for both adversarial examples and counterfactuals for many datapoints.

<img src="plots/distances.png" width="300" />

