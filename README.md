# Improving Policy Optimization with Generalist-Specialist Learning

This is the official repository for the RL framework called **GSL** presented in the following paper:

### **Improving Policy Optimization with Generalist-Specialist Learning**<br>
Zhiwei Jia, Xuanlin Li, Zhan Ling, Shuang Liu, Yiran Wu, Hao Su<br>
UC San Diego<br>
ICML 2022<br>


<p align="center">
  <img src='teaser.png' width="400"/><br>
  <a href="https://arxiv.org/abs/2206.12984">[arXiv]</a>&emsp;<a href="https://zjia.eng.ucsd.edu/gsl">[website]</a>
</p>

Generalist-specialist learning (GSL) is a meta-algorithm for large-scale policy learning.
We empirically observe that an agent trained on many variations (a **generalist**) tends to learn faster at the beginning, yet its performance plateaus at a less optimal level for a long time. 
In contrast, an agent trained only on a few variations (a **specialist**) can often achieve high returns under a limited computational budget. 
GSL is an effort to have the best of both worlds by combining generalist and specialist learning in a well-principled three-stage framework. 
We show that GSL pushes the envelope of policy learning on several challenging and popular benchmarks including [Procgen](https://openai.com/blog/procgen-benchmark/), [Meta-World](https://meta-world.github.io/) and [ManiSkill](https://sapien.ucsd.edu/challenges/maniskill2021/).

As a meta-algorithm, GSL can potentially work with any (actor-critic style) policy learning algorithms. 
As illustrated below, GSL is straightforward and easy to adapt to any modern RL frameworks.

<p align="center">
  <img src='algorithm.PNG' width="400"/><br>
</p>
