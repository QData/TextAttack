Lessons learned in designing TextAttack
=========================================


*This documentation page was adapted from [Our Workshop Paper in EMNLP 2nd Workshop for Natural Language Processing Open Source Software (NLP-OSS)](https://arxiv.org/abs/2010.01724).*


TextAttack is an open-source Python toolkit for adversarial attacks, adversarial training, and data augmentation in NLP. TextAttack unites 15+ papers from the NLP adversarial attack literature into a single shared framework, with many components reused across attacks. This framework allows both researchers and developers to test and study the weaknesses of their NLP models. 

## Challenges in Design


One of the challenges for building such tools is that the tool should be flexible enough to work with many different deep learning frameworks (e.g. PyTorch, Tensorflow, Scikit-learn). Also, the tool should be able to work with datasets from various sources and in various formats. Lastly, the tools needs to be compatible with different hardware setups. 


## Our design tips 

We provide the following broad advice to help other future developers create user-friendly NLP libraries in Python:
- To become model-agnostic, implement a model wrapper class: a model is anything that takes string input(s) and returns a prediction.
- To become data-agnostic, take dataset inputs as (input, output) pairs, where each model input is represented as an OrderedDict.
- Do not plan for inputs (tensors, lists, etc.) to be a certain size or shape unless explicitly necessary.
- Centralize common text operations, like parsing and string-level operations, in one class.
- Whenever possible, cache repeated computations, including model inferences.
- If your program runs on a single GPU, but your system contains $N$ GPUs, you can obtain an performance boost proportional to N through parallelism.
- Dynamically choose between devices. (Do not require a GPU or TPU if one is not necessary.)


 Our modular and extendable design allows us to reuse many components to offer 15+ different adversarial attack methods proposed by literature. Our model-agnostic and dataset-agnostic design allows users to easily run adversarial attacks against their own models built using any deep learning framework. We hope that our lessons from developing TextAttack will help others create user-friendly open-source NLP libraries.


## TextAttack flowchart

![TextAttack flowchart](/_static/imgs/intro/textattack_components.png)


+ Here is a summary diagram of TextAttack Ecosystem

![diagram](/_static/imgs/intro/textattack_ecosystem.png)



## More Details in Reference

```
@misc{morris2020textattack,
      title={TextAttack: Lessons learned in designing Python frameworks for NLP}, 
      author={John X. Morris and Jin Yong Yoo and Yanjun Qi},
      year={2020},
      eprint={2010.01724},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

