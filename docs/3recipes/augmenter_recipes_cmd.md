# Augmenter Recipes CommandLine Use 

Transformations and constraints can be used for simple NLP data augmentations. 

The [`examples/`](https://github.com/QData/TextAttack/tree/master/examples) folder includes scripts showing common TextAttack usage for training models, running attacks, and augmenting a CSV file. 

The [documentation website](https://textattack.readthedocs.io/en/latest) contains walkthroughs explaining basic usage of TextAttack, including building a custom transformation and a custom constraint..


### Augmenting Text: `textattack augment`

Many of the components of TextAttack are useful for data augmentation. The `textattack.Augmenter` class
uses a transformation and a list of constraints to augment data. We also offer  built-in recipes
for data augmentation:
- `wordnet` augments text by replacing words with WordNet synonyms
- `embedding` augments text by replacing words with neighbors in the counter-fitted embedding space, with a constraint to ensure their cosine similarity is at least 0.8
- `charswap` augments text by substituting, deleting, inserting, and swapping adjacent characters
- `eda` augments text with a combination of word insertions, substitutions and deletions.
- `checklist` augments text by contraction/extension and by substituting names, locations, numbers.
- `clare` augments text by replacing, inserting, and merging with a pre-trained masked language model.


### Augmentation Command-Line Interface
The easiest way to use our data augmentation tools is with `textattack augment <args>`. 

`textattack augment`
takes an input CSV file, the "text" column to augment, along with the number of words to change per augmentation
and the number of augmentations per input example. It outputs a CSV in the same format with all the augmented examples in the proper columns.

> For instance, when given the following as `examples.csv`:

```
"text",label
"the rock is destined to be the 21st century's new conan and that he's going to make a splash even greater than arnold schwarzenegger , jean- claud van damme or steven segal.", 1
"the gorgeously elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .", 1
"take care of my cat offers a refreshingly different slice of asian cinema .", 1
"a technically well-made suspenser . . . but its abrupt drop in iq points as it races to the finish line proves simply too discouraging to let slide .", 0
"it's a mystery how the movie could be released in this condition .", 0
```

The command
```
textattack augment --input-csv examples.csv --output-csv output.csv  --input-column text --recipe embedding --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original
```
will augment the `text` column by altering 10% of each example's words, generating twice as many augmentations as original inputs, and exclude the original inputs from the
output CSV. (All of this will be saved to `augment.csv` by default.)

> **Tip:** Just as running attacks interactively, you can also pass `--interactive` to augment samples inputted by the user to quickly try out different augmentation recipes!


After augmentation, here are the contents of `augment.csv`:
```
text,label
"the rock is destined to be the 21st century's newest conan and that he's gonna to make a splashing even stronger than arnold schwarzenegger , jean- claud van damme or steven segal.",1
"the rock is destined to be the 21tk century's novel conan and that he's going to make a splat even greater than arnold schwarzenegger , jean- claud van damme or stevens segal.",1
the gorgeously elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of expression significant adequately describe co-writer/director pedro jackson's expanded vision of j . rs . r . tolkien's middle-earth .,1
the gorgeously elaborate continuation of 'the lordy of the piercings' trilogy is so huge that a column of mots cannot adequately describe co-novelist/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .,1
take care of my cat offerings a pleasantly several slice of asia cinema .,1
taking care of my cat offers a pleasantly different slice of asiatic kino .,1
a technically good-made suspenser . . . but its abrupt drop in iq points as it races to the finish bloodline proves straightforward too disheartening to let slide .,0
a technically well-made suspenser . . . but its abrupt drop in iq dot as it races to the finish line demonstrates simply too disheartening to leave slide .,0
it's a enigma how the film wo be releases in this condition .,0
it's a enigma how the filmmaking wo be publicized in this condition .,0
```

The 'embedding' augmentation recipe uses counterfitted embedding nearest-neighbors to augment data.

