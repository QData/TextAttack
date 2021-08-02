TextAttack Extended Functions (Multilingual)
============================================



## Multilingual Supports

- see example code: [https://github.com/QData/TextAttack/blob/master/examples/attack/attack_camembert.py](https://github.com/QData/TextAttack/blob/master/examples/attack/attack_camembert.py) for using our framework to attack French-BERT. 

- see tutorial notebook: [https://textattack.readthedocs.io/en/latest/2notebook/Example_4_CamemBERT.html](https://textattack.readthedocs.io/en/latest/2notebook/Example_4_CamemBERT.html) for using our framework to attack French-BERT. 




## We have built a new WebDemo For Visulizing TextAttack generated Examples; 

- [TextAttack-WebDemo Github](https://github.com/QData/TextAttack-WebDemo)



## User defined custom inputs and models 


### Custom Datasets:  Dataset from a file

Loading a dataset from a file is very similar to loading a model from a file. A 'dataset' is any iterable of `(input, output)` pairs.
The following example would load a sentiment classification dataset from file `my_dataset.py`:

```python
dataset = [('Today was....', 1), ('This movie is...', 0), ...]
```

You can then run attacks on samples from this dataset by adding the argument `--dataset-from-file my_dataset.py`.


#### Custom Model:  from a file
To experiment with a model you've trained, you could create the following file
and name it `my_model.py`:

```python
model = load_your_model_with_custom_code() # replace this line with your model loading code
tokenizer = load_your_tokenizer_with_custom_code() # replace this line with your tokenizer loading code
```

Then, run an attack with the argument `--model-from-file my_model.py`. The model and tokenizer will be loaded automatically.



### Custom attack components 

The [documentation website](https://textattack.readthedocs.io/en/latest) contains walkthroughs explaining basic usage of TextAttack, including building a custom transformation and a custom constraint..
