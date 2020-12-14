Four Components of TextAttack Attacks
========================================

To unify adversarial attack methods into one system, We formulate an attack as consisting of four components: a **goal function** which determines if the attack has succeeded, **constraints** defining which perturbations are valid, a **transformation** that generates potential modifications given an input, and a **search method** which traverses through the search space of possible perturbations.  The attack attempts to perturb an input text such that the model output fulfills the goal function (i.e., indicating whether the attack is successful) and the perturbation adheres to the set of constraints (e.g., grammar constraint, semantic similarity constraint). A search method is used to find a sequence of transformations that produce a successful adversarial example.



This modular design enables us to easily assemble attacks from the literature while re-using components that are shared across attacks. TextAttack provides clean, readable implementations of 16 adversarial attacks from the literature. For the first time, these attacks can be benchmarked, compared, and analyzed in a standardized setting.


- Two examples showing four components of two SOTA attacks
![two-categorized-attacks](/_static/imgs/intro/01-categorized-attacks.png)





### Goal Functions

A `GoalFunction` takes as input an `AttackedText` object, scores it, and determines whether the attack has succeeded, returning a `GoalFunctionResult`.

### Constraints

A `Constraint` takes as input a current `AttackedText`, and a list of transformed `AttackedText`s. For each transformed option, it returns a boolean representing whether the constraint is met.

### Transformations

A `Transformation` takes as input an `AttackedText` and returns a list of possible transformed `AttackedText`s. For example, a transformation might return all possible synonym replacements.

### Search Methods

A `SearchMethod` takes as input an initial `GoalFunctionResult` and returns a final `GoalFunctionResult` The search is given access to the `get_transformations` function, which takes as input an `AttackedText` object and outputs a list of possible transformations filtered by meeting all of the attack’s constraints. A search consists of successive calls to `get_transformations` until the search succeeds (determined using `get_goal_results`) or is exhausted.



### On Benchmarking Attack Recipes

- Please read our analysis paper: Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples at [EMNLP BlackBoxNLP](https://arxiv.org/abs/2009.06368). 

- As we emphasized in the above paper, we don't recommend to directly compare Attack Recipes out of the box. 

- This is due to that attack recipes in the recent literature used different ways or thresholds in setting up their constraints. Without the constraint space held constant, an increase in attack success rate could come from an improved search or a better transformation method or a less restrictive search space. 



### Four components in Attack Recipes we have implemented 


- TextAttack provides clean, readable implementations of 16 adversarial attacks from the literature.

- To run an attack recipe: `textattack attack --recipe [recipe_name]`



<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th style="text-align: left;"><strong>Attack Recipe Name</strong></th>
<th style="text-align: left;"><strong>Goal Function</strong></th>
<th style="text-align: left; width:130px" ><strong>Constraints-Enforced</strong></th>
<th style="text-align: left;"><strong>Transformation</strong></th>
<th style="text-align: left;"><strong>Search Method</strong></th>
<th style="text-align: left;"><strong>Main Idea</strong></th>
</tr>
</thead>
<tbody>
  <tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on classification tasks, like sentiment classification and entailment:<br></strong></td></tr>

<tr class="even">
<td style="text-align: left;"><code>alzantot</code>  <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td style="text-align: left;"><sub>Untargeted {Classification, Entailment}</sub></td>
<td style="text-align: left;"><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td style="text-align: left;"><sub>Counter-fitted word embedding swap</sub></td>
<td style="text-align: left;"><sub>Genetic Algorithm</sub></td>
<td ><sub>from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998))</sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>bae</code> <span class="citation" data-cites="garg2020bae"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"><sub>USE sentence encoding cosine similarity</sub></td>
<td style="text-align: left;"><sub>BERT Masked Token Prediction</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>BERT masked language model transformation attack from (["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019)](https://arxiv.org/abs/2004.01970)). </td>
</tr>
<tr class="even">
<td style="text-align: left;"><code>bert-attack</code> <span class="citation" data-cites="li2020bertattack"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</td>
<td style="text-align: left;"><sub>USE sentence encoding cosine similarity, Maximum number of words perturbed</td>
<td style="text-align: left;"><sub>BERT Masked Token Prediction (with subword expansion)</td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub> (["BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al., 2020)](https://arxiv.org/abs/2004.09984))</sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>checklist</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td style="text-align: left;"><sub>{Untargeted, Targeted} Classification</sub></td>
<td style="text-align: left;"><sub>checklist distance</sub></td>
<td style="text-align: left;"><sub>contract, extend, and substitutes name entities</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>Invariance testing implemented in CheckList . (["Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (Ribeiro et al., 2020)](https://arxiv.org/abs/2005.04118))</sub></td>
</tr>
<tr class="even">
<td style="text-align: left;"> <code>clare (*coming soon*)</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td style="text-align: left;"><sub>Untargeted {Classification, Entailment}</sub></td>
<td style="text-align: left;"><sub>RoBERTa masked language model</sub></td>
<td style="text-align: left;"><sub>word swap, insertion, and merge</sub></td>
<td style="text-align: left;"><sub>Greedy</sub></td>
<td ><sub>["Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)](https://arxiv.org/abs/2009.07502))</sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>deepwordbug</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td style="text-align: left;"><sub>{Untargeted, Targeted} Classification</sub></td>
<td style="text-align: left;"><sub>Levenshtein edit distance</sub></td>
<td style="text-align: left;"><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)</sub></td>
</tr>
<tr class="even">
<td style="text-align: left;"> <code>fast-alzantot</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td style="text-align: left;"><sub>Untargeted {Classification, Entailment}</sub></td>
<td style="text-align: left;"><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td style="text-align: left;"><sub>Counter-fitted word embedding swap</sub></td>
<td style="text-align: left;"><sub>Genetic Algorithm</sub></td>
<td ><sub>Modified, faster version of the Alzantot et al. genetic algorithm, from (["Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)](https://arxiv.org/abs/1909.00986))</sub></td>
</tr>
<tr class="even">
<td style="text-align: left;"><code>hotflip</code> (word swap) <span class="citation" data-cites="Ebrahimi2017HotFlipWA"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"><sub>Word Embedding Cosine Similarity, Part-of-speech match, Number of words perturbed</sub></td>
<td style="text-align: left;"><sub>Gradient-Based Word Swap</sub></td>
<td style="text-align: left;"><sub>Beam search</sub></td>
<td ><sub> (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751))</sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>iga</code> <span class="citation" data-cites="iga-wang2019natural"></span></td>
<td style="text-align: left;"><sub>Untargeted {Classification, Entailment}</sub></td>
<td style="text-align: left;"><sub>Percentage of words perturbed, Word embedding distance</sub></td>
<td style="text-align: left;"><sub>Counter-fitted word embedding swap</sub></td>
<td style="text-align: left;"><sub>Genetic Algorithm</sub></td>
<td ><sub>Improved genetic algorithm -based word substitution from (["Natural Language Adversarial Attacks and Defenses in Word Level (Wang et al., 2019)"](https://arxiv.org/abs/1909.06723)</sub></td>
</tr>
<tr class="even">
<td style="text-align: left;"><code>input-reduction</code> <span class="citation" data-cites="feng2018pathologies"></span></td>
<td style="text-align: left;"><sub>Input Reduction</sub></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"><sub>Word deletion</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking , Reducing the input while maintaining the prediction through word importance ranking (["Pathologies of Neural Models Make Interpretation Difficult" (Feng et al., 2018)](https://arxiv.org/pdf/1804.07781.pdf))</sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>kuleshov</code> <span class="citation" data-cites="Kuleshov2018AdversarialEF"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"><sub>Thought vector encoding cosine similarity, Language model similarity probability</sub></td>
<td style="text-align: left;"><sub>Counter-fitted word embedding swap</sub></td>
<td style="text-align: left;"><sub>Greedy word swap</sub></td>
<td ><sub>(["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)) </sub></td>
</tr>
<tr class="even">
<td style="text-align: left;"><code>pruthi</code> <span class="citation" data-cites="pruthi2019combating"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"><sub>Minimum word length, Maximum number of words perturbed</sub></td>
<td style="text-align: left;"><sub>{Neighboring Character Swap, Character Deletion, Character Insertion, Keyboard-Based Character Swap}</sub></td>
<td style="text-align: left;"><sub>Greedy search</sub></td>
<td ><sub>simulates common typos (["Combating Adversarial Misspellings with Robust Word Recognition" (Pruthi et al., 2019)](https://arxiv.org/abs/1905.11268) </sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>pso</code> <span class="citation" data-cites="pso-zang-etal-2020-word"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"><sub>HowNet Word Swap</sub></td>
<td style="text-align: left;"><sub>Particle Swarm Optimization</sub></td>
<td ><sub>(["Word-level Textual Adversarial Attacking as Combinatorial Optimization" (Zang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.540/)) </sub></td>
</tr>
<tr class="even">
<td style="text-align: left;"><code>pwws</code> <span class="citation" data-cites="pwws-ren-etal-2019-generating"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"><sub>WordNet-based synonym swap</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR (saliency)</sub></td>
<td ><sub>Greedy attack with word importance ranking based on word saliency and synonym swap scores (["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/))</sub> </td>
</tr>
<tr class="even">
<td style="text-align: left;"><code>textbugger</code> : (black-box) <span class="citation" data-cites="Li2019TextBuggerGA"></span></td>
<td style="text-align: left;"><sub>Untargeted Classification</sub></td>
<td style="text-align: left;"><sub>USE sentence encoding cosine similarity</sub></td>
<td style="text-align: left;"><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).</sub></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><code>textfooler</code> <span class="citation" data-cites="Jin2019TextFooler"></span></td>
<td style="text-align: left;"><sub>Untargeted {Classification, Entailment}</sub></td>
<td style="text-align: left;"><sub>Word Embedding Distance, Part-of-speech match, USE sentence encoding cosine similarity</sub></td>
<td style="text-align: left;"><sub>Counter-fitted word embedding swap</sub></td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking  (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))</sub> </td>
</tr>

<tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on sequence-to-sequence models: <br></strong></td></tr>

<tr class="odd">
<td style="text-align: left;"><code>morpheus</code> <span class="citation" data-cites="morpheus-tan-etal-2020-morphin"></span></td>
<td style="text-align: left;"><sub>Minimum BLEU Score</sub> </td>
<td style="text-align: left;"></td>
<td style="text-align: left;"><sub>Inflection Word Swap</sub> </td>
<td style="text-align: left;"><sub>Greedy search</sub> </td>
<td ><sub>Greedy to replace words with their inflections with the goal of minimizing BLEU score (["It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations"](https://www.aclweb.org/anthology/2020.acl-main.263.pdf)</sub> </td>
</tr>

</tr>
<tr class="odd">
<td style="text-align: left;"><code>seq2sick</code> :(black-box) <span class="citation" data-cites="cheng2018seq2sick"></span></td>
<td style="text-align: left;"><sub>Non-overlapping output</sub> </td>
<td style="text-align: left;"></td>
<td style="text-align: left;"><sub>Counter-fitted word embedding swap</sub> </td>
<td style="text-align: left;"><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with goal of changing every word in the output translation. Currently implemented as black-box with plans to change to white-box as done in paper (["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples" (Cheng et al., 2018)](https://arxiv.org/abs/1803.01128)) </sub>  </td>
</tr>


</tbody>
</font>
</table>



- Citations

```
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Jake Grigsby and Di Jin and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
