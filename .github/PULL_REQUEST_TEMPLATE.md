# What does this PR do?

## Checklist
- [ ] The title of your pull request should be a summary of its contribution.
- [ ] Please write detailed description of what parts have been newly added and what parts have been modified. Please also explain why certain changes were made.
- [ ] If your pull request addresses an issue, please mention the issue number in the pull request description to make sure they are linked (and people consulting the issue know you   are working on it)
- [ ] To indicate a work in progress please mark it as a draft on Github.
- [ ] Make sure existing tests pass.
- [ ] Add relevant tests. No quality testing = no merge.
- [ ] All public methods must have informative docstrings that work nicely with sphinx. For new modules/files, please add/modify the appropriate `.rst` file in `TextAttack/docs/apidoc`.'


## Example:
```
This PR adds [CLARE](https://arxiv.org/abs/2009.07502) attack, which uses distilled RoBERTa masked language model to perform word swaps, word insertions, word merges (which is where we combine two adjacent words and replace it with another word) in a greedy manner. 

New additions:
- `clare` recipe as `textattack.attack_recipes.CLARE2020`.
- `WordInsertionMaskedLM`, which is similar to `WordSwapMaskedLM` but instead insert new words using masked language model.
- `WordMergeMasedLM`, which "merges" two words by replacing two adjacent words with a single word using a masked language model.
    Hand-crafted rules based on words' part-of-speeches are used to determine which two adjacent words can be merged.
-  Base classes `WordInsertion` and `WordMerge` for new categories of word transformations.

Changes:
- `WordSwapMaskedLM` has been updated to have a minimum confidence score cutoff and batching has been added for faster performance.
- Changes in directory structure for transformations for cbetter organization. Word swaps have been grouped together under `textattack.transformations.word_swaps`
    and other methods such as insertions and merges have been organized similarily. 
```
