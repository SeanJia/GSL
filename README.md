# Improving Policy Optimization with Generalist-Specialist Learning

This is the official repository for the RL framework called **GSL** presented in the following paper:

### **[Improving Policy Optimization with Generalist-Specialist Learning](https://zjia.eng.ucsd.edu/gsl)**<br>
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
We show the pseudocode (extracted from the main paper) as below.
Notice that GSL is relatively straightforward and easy to adapt to any modern RL frameworks.
In fact, in our experiments, we adapt GSL for PPO from [PFRL](https://github.com/pfnet/pfrl) on Procgen, PPG from [OpenAI-PPG](https://github.com/openai/phasic-policy-gradient) on Procgen, PPO from [garage](https://github.com/rlworkgroup/garage) for Meta-World, and SAC from [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn) for ManiSkill.
For better clarity, here in this repo, we only take [garage](https://github.com/rlworkgroup/garage) (which provides the official PPO impl for experiments in Meta-World) as an example to demonstrate the key steps in adapting the GSL framework.

<p align="center">
  <img src='algorithm.PNG' width="800"/><br>
  Pseudocode of GSL
</p>

In this example, we use PPO and DAPG as the building blocks for GSL.
The first 8 steps of GSL are straightforward (phase I generalist learning). 
Starting from step 9, it launches speciaslist training (in parallel) and collect demonstrations in step 12.  
Normally, a modern RL framework provides an encapsulation for experience collected in an episode (e.g.., used in a replay buffer).
Here in garage it refers to the `EpisodeBatch` class ([link](https://github.com/rlworkgroup/garage/blob/f056fb8f6226c83d340c869e0d5312d61acf07f0/src/garage/_dtypes.py#L455)).
When we rollout the policy and store the demos, we utilize such encapsulation and add the following code as a class function to `class VPG` in [this]() file:
```Python
    def store_demos(self, trainer, num_epochs, batch_size, path):
        """Obtain samples and store the demos.

        Args:
            trainer (Trainer): the trainer of the RL algorithm.
            num_epochs (int): the number of epochs of data to be collected.
            batch_size (int): the batch size of the episode to be collected.
            path (str): the output path for the demos to be stored.
        """
        trainer._train_args.batch_size = batch_size
        for i in range(num_epochs):
            eps = trainer.obtain_episodes(0)
            with open(f'{path}/{i}.pkl', 'wb') as f:
                pickle.dump(eps, f)
```
Then we call `trainer.store_demos(...)`, which in turn calls `VPG.store_demos(...)` to collect the demos.
We provide an example code in `demo_mt10.py`.
