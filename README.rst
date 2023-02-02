This repository is associated with the paper "Improving Neural Topic Models with Wasserstein Knowledge Distillation", accepted at ECIR 2023.
=======
CTMKD
=======
**CTMKD** is a inter-VAE knowledge distillation framework where the teacher is a `CombinedTM`_. and student is a `ZeroShotTM`_. In particular, the proposed distillation objective is to minimize the cross-entropy of the soft labels produced by the teacher and the student models, as well as to minimize the squared 2-Wasserstein distance between the latent distributions learned by the two models.

.. _CombinedTM: https://aclanthology.org/2021.acl-short.96/
.. _ZeroShotTM: https://aclanthology.org/2021.eacl-main.143/

The implemention will be avilable soon.
