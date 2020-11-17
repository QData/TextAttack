How to Cite TextAttack  
===========================

## Main Paper:  TextAttack: A Framework for Adversarial Attacks in Natural Language Processing

- Paper [EMNLP Demo](https://arxiv.org/abs/2005.05909)

- Abstract: TextAttack is a library for generating natural language adversarial examples to fool natural language processing (NLP) models. TextAttack builds attacks from four components: a search method, goal function, transformation, and a set of constraints. Researchers can use these components to easily assemble new attacks. Individual components can be isolated and compared for easier ablation studies. TextAttack currently supports attacks on models trained for text classification and entailment across a variety of datasets. Additionally, TextAttack's modular design makes it easily extensible to new NLP tasks, models, and attack strategies. 

### Our Github on TextAttack: `TextAttack <https://github.com/QData/TextAttack>`_

- Citations

```
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks in Natural Language Processing},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## Our Analysis paper: Reevaluating Adversarial Examples in Natural Language

- Paper [EMNLP Findings](https://arxiv.org/abs/2004.14174)

- Abstract:  State-of-the-art attacks on NLP models lack a shared definition of a what constitutes a successful attack. We distill ideas from past work into a unified framework: a successful natural language adversarial example is a perturbation that fools the model and follows some linguistic constraints. We then analyze the outputs of two state-of-the-art synonym substitution attacks. We find that their perturbations often do not preserve semantics, and 38% introduce grammatical errors. Human surveys reveal that to successfully preserve semantics, we need to significantly increase the minimum cosine similarities between the embeddings of swapped words and between the sentence encodings of original and perturbed sentences.With constraints adjusted to better preserve semantics and grammaticality, the attack success rate drops by over 70 percentage points.

### Our Github on Reevaluation: `Reevaluating-NLP-Adversarial-Examples Github <https://github.com/QData/Reevaluating-NLP-Adversarial-Examples>`__ 

- Citations
```
@misc{morris2020reevaluating,
      title={Reevaluating Adversarial Examples in Natural Language}, 
      author={John X. Morris and Eli Lifland and Jack Lanchantin and Yangfeng Ji and Yanjun Qi},
      year={2020},
      eprint={2004.14174},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Our Analysis paper: Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples

- Paper [EMNLP BlackNLP](https://arxiv.org/abs/2009.06368)

- Abstract:  We study the behavior of several black-box search algorithms used for generating adversarial examples for natural language processing (NLP) tasks. We perform a fine-grained analysis of three elements relevant to search: search algorithm, search space, and search budget. When new search methods are proposed in past work, the attack search space is often modified alongside the search method. Without ablation studies benchmarking the search algorithm change with the search space held constant, an increase in attack success rate could from an improved search method or a less restrictive search space. Additionally, many previous studies fail to properly consider the search algorithms' run-time cost, which is essential for downstream tasks like adversarial training. Our experiments provide a reproducible benchmark of search algorithms across a variety of search spaces and query budgets to guide future research in adversarial NLP. Based on our experiments, we recommend greedy attacks with word importance ranking when under a time constraint or attacking long inputs, and either beam search or particle swarm optimization otherwise. 

### Our Github on benchmarking:  `TextAttack-Search-Benchmark Github <https://github.com/QData/TextAttack-Search-Benchmark>`__ 


- Citations: 
```
@misc{yoo2020searching,
      title={Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples}, 
      author={Jin Yong Yoo and John X. Morris and Eli Lifland and Yanjun Qi},
      year={2020},
      eprint={2009.06368},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



## A summary diagram of TextAttack Ecosystem

![diagram](/_static/imgs/intro/textattack_ecosystem.png)

