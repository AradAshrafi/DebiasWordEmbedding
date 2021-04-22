### Mitigating Gender Bias from Word Embedding

In this project we did a re-implementation of Hard Debias Algorithm and Double-Hard Debias Algorithm ([1], [2]).
These algorithms use some hand-engineered rules based on their authors' intuition to solve the bias issue in the embeddings.
We came up with a novel idea of applying Evolutionary Algorithm to tackle this issue. 

Work's done in this repo:
- [x] Reimplementation of Hard Debias
- [x] Reimplementation of Double-Hard Debias Algorithm
- [x] Implementation of our novel Evolutionary Strategy algorithm
- [x] Evaluation of our debiased embeddings based on 4 different metrics



##References

[1] T. Bolukbasi, K.-W. Chang, J. Y. Zou, V. Saligrama, and A. T. Kalai, “Man is to computer
programmer as woman is to homemaker? debiasing word embeddings,” in Advances in
Neural Information Processing Systems, D. Lee, M. Sugiyama, U. Luxburg, I. Guyon,
and R. Garnett, Eds., vol. 29. Curran Associates, Inc., 2016. [Online]. Available: https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf

[2] T. Wang, X. V. Lin, N. F. Rajani, B. McCann, V. Ordonez, and C. Xiong, “Double-hard
debias: Tailoring word embeddings for gender bias mitigation,” in Proceedings of
the 58th Annual Meeting of the Association for Computational Linguistics. Online:
Association for Computational Linguistics, Jul. 2020, pp. 5443–5453. [Online]. Available:
https://www.aclweb.org/anthology/2020.acl-main.484 