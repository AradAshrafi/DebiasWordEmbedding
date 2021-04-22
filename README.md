## Mitigating Gender Bias from Word Embedding

#### What it this task about? Why it is important?
Fairness in machine learning systems has been recently attracting much attention from academic researchers, industry, and the public. An important aspect of fairness in machine learning is bias in the machine learning models. This bias can arise due to both the data and algorithms

Word embeddings will map each word into a vector space in order to make it usable for further text analysis and machine learning algorithms. These vector representations encode meaning and hold the semantic information of words so that words with similar meanings are close to each other in the vector space and the differences between these embeddings represent relationships between words. However, word embeddings are biased because they are trained on text corpora that contain direct and indirect human biases. We focus on mitigating one particular type of bias: gender bias. 

Word embeddings tend to maintain the gender stereotypes that pervade language and this bias can be magnified when machine learning models are trained using these word embeddings and then applied to downstream tasks.

&nbsp; </br>
&nbsp; </br>
&nbsp; </br>

#### What is done?
In this project we did a re-implementation of Hard Debias Algorithm and Double-Hard Debias Algorithm ([1], [2]). Those algorithms deal with gender bias and proposed some novel approaches to address that.
Hard Debias and Double Hard Debias algorithms use some hand-engineered rules based on their authors' intuition to solve the bias issue in the embeddings.
We came up with a novel idea of applying Evolutionary Algorithm to tackle this issue. 

Work's done in this repo:
- [x] Reimplementation of Hard Debias
- [x] Reimplementation of Double-Hard Debias Algorithm
- [x] Implementation of our novel Evolutionary Strategy algorithm
- [x] Evaluation of our debiased embeddings based on 4 different metrics

#### Repository's Details

The implementation of three different algorithms alongwith Evaluations on Word2Vec and Glove could be find in this repo.
* [hard_debias.py](https://github.com/AradAshrafi/DebiasWordEmbedding/blob/main/hard_debias.py): Implementation of hard debias. By running hard_debias function and giving its necessary inputs it will do the job.
* [double_hard_debias.py](https://github.com/AradAshrafi/DebiasWordEmbedding/blob/main/double_hard_debias.py): Implementation of double hard debias. Instruction on running it is similar to above.
* [evolutionary_algorithm.ipynb](https://github.com/AradAshrafi/DebiasWordEmbedding/blob/main/evolutionary_algorithm.ipynb): Implementation of evolutionary debias. Instruction on running it is inside the jupyter notebook itself.
* [Glove Evaluation](https://github.com/AradAshrafi/DebiasWordEmbedding/blob/main/Evaluation/evaluation_GloVe.ipynb): A Jupyter notebook which contains the result of the Evaluation process on Glove embedding.
* Results of running our Evolutionary algorithm in different scenarios could be find in the Jupyter notebooks starting with "evolutionary_algorithm" and "ea" located in the root directory.

#### Requirements

Several evaluation tests rely on installing and importing the Word Embedding Benchmarks repository.

[Word Embedding Benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks)

## References

[1] T. Bolukbasi, K.-W. Chang, J. Y. Zou, V. Saligrama, and A. T. Kalai, “Man is to computer
programmer as woman is to homemaker? debiasing word embeddings,” in Advances in
Neural Information Processing Systems, D. Lee, M. Sugiyama, U. Luxburg, I. Guyon,
and R. Garnett, Eds., vol. 29. Curran Associates, Inc., 2016. [Online]. Available: https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf

[2] T. Wang, X. V. Lin, N. F. Rajani, B. McCann, V. Ordonez, and C. Xiong, “Double-hard
debias: Tailoring word embeddings for gender bias mitigation,” in Proceedings of
the 58th Annual Meeting of the Association for Computational Linguistics. Online:
Association for Computational Linguistics, Jul. 2020, pp. 5443–5453. [Online]. Available:
https://www.aclweb.org/anthology/2020.acl-main.484 
