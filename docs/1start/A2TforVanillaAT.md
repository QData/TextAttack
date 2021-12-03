Making Vanilla Adversarial Training of NLP Models Feasible!
=========================================================================


*This documentation page was adapted from Our  2021 EMNLP Findings paper: [Towards Improving Adversarial Training of NLP Models](https://arxiv.org/abs/2109.00544).*


### Title: Towards Improving Adversarial Training of NLP Models


- Abstract:  Adversarial training, a method for learning robust deep neural networks, constructs adversarial examples during training. However, recent methods for generating NLP adversarial examples involve combinatorial search and expensive sentence encoders for constraining the generated instances. As a result, it remains challenging to use vanilla adversarial training to improve NLP models' performance, and the benefits are mainly uninvestigated. This paper proposes a simple and improved vanilla adversarial training process for NLP models, which we name Attacking to Training (A2T). The core part of A2T is a new and cheaper word substitution attack optimized for vanilla adversarial training. We use A2T to train BERT and RoBERTa models on IMDB, Rotten Tomatoes, Yelp, and SNLI datasets. Our results empirically show that it is possible to train robust NLP models using a much cheaper adversary. We demonstrate that vanilla adversarial training with A2T can improve an NLP model's robustness to the attack it was originally trained with and also defend the model against other types of word substitution attacks. Furthermore, we show that A2T can improve NLP models' standard accuracy, cross-domain generalization, and interpretability. 


#### Video recording of this talk: [https://underline.io/events/192/sessions/7928/lecture/38377-towards-improving-adversarial-training-of-nlp-models](https://underline.io/events/192/sessions/7928/lecture/38377-towards-improving-adversarial-training-of-nlp-models)

### Code is available 

We share all codes of this analysis at [https://github.com/QData/Textattack-A2T](https://github.com/QData/Textattack-A2T) .


### Citations: 
```
@misc{yoo2021improving,
      title={Towards Improving Adversarial Training of NLP Models}, 
      author={Jin Yong Yoo and Yanjun Qi},
      year={2021},
      eprint={2109.00544},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



### A2T Attack Recipe 

- We add A2T as part of textattack at [https://github.com/QData/TextAttack/blob/master/textattack/attack_recipes/a2t_yoo_2021.py](https://github.com/QData/TextAttack/blob/master/textattack/attack_recipes/a2t_yoo_2021.py)