CartPole results
================

This directory contains results of the CartPole experiments.

- Raw data are stored in the ``raw`` directory as the outputs logged by the simulator.

  - For ``symmetry`` subdir the base model is the one from the commit ``d96ba8d`` and:

    - ``base`` vs. ``sym`` name prefix denote whether the symmetry was used in the DQN.
    - ``D-x-y-z`` denote that the numbers of units in the dense layers are consequently
    ``x``, ``y`` and ``z``.
    - ``no-BN`` denotes that the batch normalization was turned off for the experiment.
    - ``ELU`` or ``tanh`` denote non-default activation function (default is ``ReLU``).
    - Examples with tag ``before-BN`` apply the activation function before batch normalization.
    - Examples with tags ``ELU``, ``tanh``, ``before-BN`` and ``all-BN`` use batch normalization as the first layer.
    - Optional ``__n`` prefix might be used to distinguish different runs under the same parameters.

  - For ``reg_symmetry`` subdir the base model is the one from the commit ``8a6f697`` and:

    - ``reg-X`` denote that the regularization weight is equal to ``1e-X``
    - ``alr-m-X-sY-eZ`` denote that adaptive learning rate is used and the exponential coefficient equals
      ``mean_duration / X``, starting learning rate equals ``1e-Y`` and ending learning rate equals ``1e-Z``.

- ``helper.py`` is the helper script for analyzing the logs.
