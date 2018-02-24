OpenAI Deep RL Experiments with PyTorch
#######################################

Usage
=====

Use Python's interpreter ``-m`` to run one of the experiments, e.g.::

    python3 -m torchai.cartpole

Conclusions
===========

Symmetry in CartPole DQN
------------------------

Using information about the symmetry of the problem should reduce the amount knowledge that should be gained by the
agent in order to acquire correct behaviour. Using this prior leads to:

- much faster solution (the model is often efficient after less than 100 episodes),
- difficulties in maintaining the solution (the solution might be both found and forgotten during first 100 episodes).

The second property makes it difficult to achieve high score. Usually we'd like to achieve the efficient model as fast
as possible and freeze the model's weights. Unfortunately this approach might lead to suboptimal solution. Actually,
it's all about stating how efficient is sufficient.

Batch Normalization
-------------------

Batch normalization usually improves performance and training time but surprisingly it seems essential in this problem.
A model without batch normalization wasn't usually able to last longer than 10 iterations in the CartPole problem.
