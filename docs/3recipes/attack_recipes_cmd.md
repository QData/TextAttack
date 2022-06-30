# Attack Recipes CommandLine Use

We provide a number of pre-built attack recipes, which correspond to attacks from the literature. 


## Help: `textattack --help`

TextAttack's main features can all be accessed via the `textattack` command. Two very
common commands are `textattack attack <args>`, and `textattack augment <args>`. You can see more
information about all commands using 
```bash
textattack --help 
```
or a specific command using, for example,
```bash
textattack attack --help
```

The [`examples/`](https://github.com/QData/TextAttack/tree/master/examples) folder includes scripts showing common TextAttack usage for training models, running attacks, and augmenting a CSV file. 


The [documentation website](https://textattack.readthedocs.io/en/latest) contains walkthroughs explaining basic usage of TextAttack, including building a custom transformation and a custom constraint..

## Running Attacks: `textattack attack --help`

The easiest way to try out an attack is via the command-line interface, `textattack attack`. 

> **Tip:** If your machine has multiple GPUs, you can distribute the attack across them using the `--parallel` option. For some attacks, this can really help performance.

Here are some concrete examples:

*TextFooler on BERT trained on the MR sentiment classification dataset*: 
```bash
textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 100
```

*DeepWordBug on DistilBERT trained on the Quora Question Pairs paraphrase identification dataset*: 
```bash
textattack attack --model distilbert-base-uncased-cola --recipe deepwordbug --num-examples 100
```

*Beam search with beam width 4 and word embedding transformation and untargeted goal function on an LSTM*:
```bash
textattack attack --model lstm-mr --num-examples 20 \
 --search-method beam-search^beam_width=4 --transformation word-swap-embedding \
 --constraints repeat stopword max-words-perturbed^max_num_words=2 embedding^min_cos_sim=0.8 part-of-speech \
 --goal-function untargeted-classification
```

> **Tip:** Instead of specifying a dataset and number of examples, you can pass `--interactive` to attack samples inputted by the user.

## Attacks and Papers Implemented ("Attack Recipes"): `textattack attack --recipe [recipe_name]`

We include attack recipes which implement attacks from the literature. You can list attack recipes using `textattack list attack-recipes`.

To run an attack recipe: `textattack attack --recipe [recipe_name]`


<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th><strong>Attack Recipe Name</strong></th>
<th><strong>Goal Function</strong></th>
<th><strong>ConstraintsEnforced</strong></th>
<th><strong>Transformation</strong></th>
<th><strong>Search Method</strong></th>
<th><strong>Main Idea</strong></th>
</tr>
</thead>
<tbody>
  <tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on classification tasks, like sentiment classification and entailment:<br></strong></td></tr>

<tr>
<td><code>alzantot</code>  <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998))</sub></td>
</tr>
<tr>
<td><code>bae</code> <span class="citation" data-cites="garg2020bae"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>BERT Masked Token Prediction</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>BERT masked language model transformation attack from (["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019)](https://arxiv.org/abs/2004.01970)). </td>
</tr>
<tr>
<td><code>bert-attack</code> <span class="citation" data-cites="li2020bertattack"></span></td>
<td><sub>Untargeted Classification</td>
<td><sub>USE sentence encoding cosine similarity, Maximum number of words perturbed</td>
<td><sub>BERT Masked Token Prediction (with subword expansion)</td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub> (["BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al., 2020)](https://arxiv.org/abs/2004.09984))</sub></td>
</tr>
<tr>
<td><code>checklist</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>checklist distance</sub></td>
<td><sub>contract, extend, and substitutes name entities</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Invariance testing implemented in CheckList . (["Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (Ribeiro et al., 2020)](https://arxiv.org/abs/2005.04118))</sub></td>
</tr>
<tr>
<td> <code>clare</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>RoBERTa Masked Prediction for token swap, insert and merge</sub></td>
<td><sub>Greedy</sub></td>
<td ><sub>["Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)](https://arxiv.org/abs/2009.07502))</sub></td>
</tr>
<tr>
<td><code>deepwordbug</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>Levenshtein edit distance</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)</sub></td>
</tr>
<tr>
<td> <code>faster-alzantot</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Modified, faster version of the Alzantot et al. genetic algorithm, from (["Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)](https://arxiv.org/abs/1909.00986))</sub></td>
</tr>
<tr>
<td><code>hotflip</code> (word swap) <span class="citation" data-cites="Ebrahimi2017HotFlipWA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Word Embedding Cosine Similarity, Part-of-speech match, Number of words perturbed</sub></td>
<td><sub>Gradient-Based Word Swap</sub></td>
<td><sub>Beam search</sub></td>
<td ><sub> (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751))</sub></td>
</tr>
<tr>
<td><code>iga</code> <span class="citation" data-cites="iga-wang2019natural"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Improved genetic algorithm -based word substitution from (["Natural Language Adversarial Attacks and Defenses in Word Level (Wang et al., 2019)"](https://arxiv.org/abs/1909.06723)</sub></td>
</tr>
<tr>
<td><code>input-reduction</code> <span class="citation" data-cites="feng2018pathologies"></span></td>
<td><sub>Input Reduction</sub></td>
<td></td>
<td><sub>Word deletion</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking , Reducing the input while maintaining the prediction through word importance ranking (["Pathologies of Neural Models Make Interpretation Difficult" (Feng et al., 2018)](https://arxiv.org/pdf/1804.07781.pdf))</sub></td>
</tr>
<tr>
<td><code>kuleshov</code> <span class="citation" data-cites="Kuleshov2018AdversarialEF"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Thought vector encoding cosine similarity, Language model similarity probability</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy word swap</sub></td>
<td ><sub>(["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)) </sub></td>
</tr>
<tr>
<td><code>pruthi</code> <span class="citation" data-cites="pruthi2019combating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Minimum word length, Maximum number of words perturbed</sub></td>
<td><sub>{Neighboring Character Swap, Character Deletion, Character Insertion, Keyboard-Based Character Swap}</sub></td>
<td><sub>Greedy search</sub></td>
<td ><sub>simulates common typos (["Combating Adversarial Misspellings with Robust Word Recognition" (Pruthi et al., 2019)](https://arxiv.org/abs/1905.11268) </sub></td>
</tr>
<tr>
<td><code>pso</code> <span class="citation" data-cites="pso-zang-etal-2020-word"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>HowNet Word Swap</sub></td>
<td><sub>Particle Swarm Optimization</sub></td>
<td ><sub>(["Word-level Textual Adversarial Attacking as Combinatorial Optimization" (Zang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.540/)) </sub></td>
</tr>
<tr>
<td><code>pwws</code> <span class="citation" data-cites="pwws-ren-etal-2019-generating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>WordNet-based synonym swap</sub></td>
<td><sub>Greedy-WIR (saliency)</sub></td>
<td ><sub>Greedy attack with word importance ranking based on word saliency and synonym swap scores (["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/))</sub> </td>
</tr>
<tr>
<td><code>textbugger</code> : (black-box) <span class="citation" data-cites="Li2019TextBuggerGA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).</sub></td>
</tr>
<tr>
<td><code>textfooler</code> <span class="citation" data-cites="Jin2019TextFooler"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Word Embedding Distance, Part-of-speech match, USE sentence encoding cosine similarity</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking  (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))</sub> </td>
</tr>

<tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on sequence-to-sequence models: <br></strong></td></tr>

<tr>
<td><code>morpheus</code> <span class="citation" data-cites="morpheus-tan-etal-2020-morphin"></span></td>
<td><sub>Minimum BLEU Score</sub> </td>
<td></td>
<td><sub>Inflection Word Swap</sub> </td>
<td><sub>Greedy search</sub> </td>
<td ><sub>Greedy to replace words with their inflections with the goal of minimizing BLEU score (["It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations"](https://www.aclweb.org/anthology/2020.acl-main.263.pdf)</sub> </td>
</tr>

</tr>
<tr>
<td><code>seq2sick</code> :(black-box) <span class="citation" data-cites="cheng2018seq2sick"></span></td>
<td><sub>Non-overlapping output</sub> </td>
<td></td>
<td><sub>Counter-fitted word embedding swap</sub> </td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with goal of changing every word in the output translation. Currently implemented as black-box with plans to change to white-box as done in paper (["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples" (Cheng et al., 2018)](https://arxiv.org/abs/1803.01128)) </sub>  </td>
</tr>


</tbody>
</font>
</table>



## Recipe Usage Examples

Here are some examples of testing attacks from the literature from the command-line:

*TextFooler against BERT fine-tuned on SST-2:*
```bash
textattack attack --model bert-base-uncased-sst2 --recipe textfooler --num-examples 10
```

*seq2sick (black-box) against T5 fine-tuned for English-German translation:*
```bash
 textattack attack --model t5-en-de --recipe seq2sick --num-examples 100
```
