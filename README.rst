=======
CTMKD
=======
**CTMKD** is a inter-VAE knowledge distillation framework where the teacher is a `CombinedTM`_. and student is a `ZeroShotTM`_. In particular, the proposed distillation objective is to minimize the cross-entropy of the soft labels produced by the teacher and the student models, as well as to minimize the squared 2-Wasserstein distance between the latent distributions learned by the two models.

.. _CombinedTM: https://aclanthology.org/2021.acl-short.96/
.. _ZeroShotTM: https://aclanthology.org/2021.eacl-main.143/

.. image:: https://github.com/AdhyaSuman/CTMKD/blob/master/misc/KD_Arch_updated_v1.png
   :align: center
   :width: 600px
   
Datasets
--------
We have used the datasets **20NewsGroup (20NG)** and **M10** from in OCTIS_.


How to cite this work?
----------------------

This work has been accepted at ECIR 2023!

Read the paper:

1. `Springer`_

2. `arXiv`_

If you decide to use this resource, please cite:

.. _`Springer`: https://link.springer.com/chapter/10.1007/978-3-031-28238-6_21

.. _`arXiv`: https://arxiv.org/abs/2303.15350


::

    @InProceedings{adhya2023improving, 
        author="Adhya, Suman and Sanyal, Debarshi Kumar",
        editor="Kamps, Jaap and Goeuriot, Lorraine and Crestani, Fabio and Maistro, Maria and Joho, Hideo and Davis, Brian and Gurrin, Cathal and Kruschwitz, Udo and Caputo, Annalina",
        title="Improving Neural Topic Models with Wasserstein Knowledge Distillation",
        booktitle="Advances in Information Retrieval",
        year="2023",
        publisher="Springer Nature Switzerland",
        address="Cham",
        pages="321--330",
        abstract="Topic modeling is a dominant method for exploring document collections on the web and in digital libraries. Recent approaches to topic modeling use pretrained contextualized language models and variational autoencoders. However, large neural topic models have a considerable memory footprint. In this paper, we propose a knowledge distillation framework to compress a contextualized topic model without loss in topic quality. In particular, the proposed distillation objective is to minimize the cross-entropy of the soft labels produced by the teacher and the student models, as well as to minimize the squared 2-Wasserstein distance between the latent distributions learned by the two models. Experiments on two publicly available datasets show that the student trained with knowledge distillation achieves topic coherence much higher than that of the original student model, and even surpasses the teacher while containing far fewer parameters than the teacher. The distilled model also outperforms several other competitive topic models on topic coherence.",
        isbn="978-3-031-28238-6"}
  

Acknowledgment
--------------
All experiments are conducted using OCTIS_ which is an integrated framework for topic modeling.

**OCTIS**: Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. (2021). `OCTIS: Comparing and Optimizing Topic models is Simple!`. EACL. https://www.aclweb.org/anthology/2021.eacl-demos.31/

.. _OCTIS: https://github.com/MIND-Lab/OCTIS
