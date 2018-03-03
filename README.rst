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

More details in ``results/cartpole/symmetry.ipnyb``.

Batch Normalization
-------------------

Batch normalization usually improves performance and training time but surprisingly it seems essential in this problem.
A model without batch normalization wasn't usually able to last longer than 10 iterations in the CartPole problem.

Adaptive Learning Rate
----------------------

Decreasing the learning rate when the score increases makes it easier to maintain satisfying behavior once it is
achieved. The mean scores achieved by using such learning rate adaptation were usually much higher since it counteracts
forgetting the correct behavior learnt for unstable states (states which lie close the the states with very low reward).

The drawback is that in this setting the model tends to overfit to certain initial states which enables it to quickly
achieve stable state. Once such state is achieved all consequent states are very similar which prevent from learning
how to behave in some corner cases. Although low learning rate prevents the agent from forgetting the correct behaviors
it has already learnt it does not help in acquiring the new ones. This is not strictly related to the adaptive learning
rate but rather to the typical phenomena of choosing exploitation over exploration.

A good example is a CartPole problem where the cart slowly moves toward one of the edges of the environment. The
movement speed tends to get slower as the DQN learns but the agent is not able to learn how to change the direction
because the need of doing so occurs very rarely. This is because the agent excessively explores the states where it's
not required.

Most of the corner case states occur during the experiments with very low frequency. Since there is no way to force the
agent to start near the more valuable unstable states and it wouldn't be desirable to make him approach them it seems
like a valid option would be to somehow change the weights of some particular examples in the replay buffer. Such weight
might for example affect the probability of selecting given example for the training batch or make them less likely to
be removed from the replay buffer (currently the replay buffer is a deterministic FIFO).

Raw logs can be found in ``results/cartpole/raw/reg_alr``.
