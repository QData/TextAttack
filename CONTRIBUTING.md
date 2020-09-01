# How can I contribute to TextAttack?

We welcome contributions from all members of the community– and there are lots
of ways to help without editing the code! Answering questions, helping others, 
reaching out and improving the documentations are immensely valuable to the 
community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

## Slack Channel

For help and realtime updates related to TextAttack, please [join the TextAttack Slack](https://join.slack.com/t/textattack/shared_invite/zt-ez3ts03b-Nr55tDiqgAvCkRbbz8zz9g)!

## Ways to contribute

There are lots of ways you can contribute to TextAttack:
* Submitting issues on Github to report bugs or make feature requests
* Fixing outstanding issues with the existing code
* Implementing new features
* Adding support for new models and datasets
* Contributing to the examples or to the documentation

*All are equally valuable to the community.*

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Found a bug?

TextAttack can remain robust and reliable thanks to users who notify us of
the problems they encounter. So thank you for [reporting an issue](https://github.com/QData/TextAttack/issues).

We also have a suite of tests intended to detect bugs before they enter the 
codebase. That said, they still happen (Turing completeness and all) so it's up
to you to report the bugs you find! We would really appreciate it if you could 
make sure the bug was not already reported (use the search bar on Github under 
Issues).

To help us fix your issue quickly, please follow these steps:

* Include your **OS type and version**, the versions of **Python**, **PyTorch** and
  **Tensorflow** when applicable;
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s;
* Provide the *full* traceback if an exception is raised.

### Do you want to add your model?

Awesome! Please provide the following information:

* Short description of the model and link to the paper;
* Link to the implementation if it is open-source;
* Link to the model weights if they are available.

If you are willing to contribute the model yourself, let us know so we can best
guide you. We can host your model on our S3 server, but if you trained your
model using `transformers`, it's better if you host your model on their 
[model hub](https://huggingface.co/models).

### Do you want a new feature: a component, a recipe, or something else?

A world-class feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain
    why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear
    about it!
  * Is it something you worked on and think could benefit the community?
    Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.


## Start contributing! (Pull Requests)

Before writing code, we strongly advise you to search through the exising PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
`textattack`. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/QData/TextAttack) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/TextAttack.git
   $ cd TextAttack
   $ git remote add upstream https://github.com/QData/TextAttack
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **do not** work on the `master` branch.

4. Set up a development environment by running the following commands in a virtual environment:

   
   ```bash
   $ cd TextAttack
   $ pip install -e . ".[dev]"
   $ pip install black isort pytest pytest-xdist
   ```
   
   This will install `textattack` in editable mode and install `black` and 
   `isort`, packages we use for code formatting.
   
   (If TextAttack was already installed in the virtual environment, remove
   it with `pip uninstall textattack` before reinstalling it in editable
   mode with the `-e` flag.)
   
5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes:

   ```bash
   $ make test
   ```
   
   (or just simply `pytest`.)
   
   > **Tip:** if you're fixing just one or two tests, you can run only the last tests that failed using `pytest --lf`.

   `textattack` relies on `black` and `isort` to format its source code
   consistently. After you make changes, format them with:

   ```bash
   $ make format
   ```

   You can run quality checks to make sure your code is formatted properly
   using this command:

   ```bash
   $ make lint
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Add documentation.
   
   Our docs are in the `docs/` folder. Thanks to `sphinx-automodule`, adding 
   documentation for a new code file should just be two lines. Our docs will 
   automatically generate from the comments you added to your code. If you're 
   adding an attack recipe, add a reference in `attack_recipes.rst`. 
   If you're adding a transformation, add a reference in `transformation.rst`, etc. 

   You can build the docs and view the updates using `make docs`. If you're 
   adding a tutorial or something where you want to update the docs multiple
   times, you can run `make docs-auto`. This will run a server using 
   `sphinx-autobuild` that should automatically reload whenever you change
   a file.

7. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

8. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution.
2. If your pull request adresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please mark it as a draft on Github.
4. Make sure existing tests pass.
5. Add relevant tests. No quality testing = no merge.
6. All public methods must have informative docstrings that work nicely with sphinx.

### Tests

You can run TextAttack tests with `pytest`. Just type `make test`.


#### This guide was heavily inspired by the awesome [transformers guide to contributing](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md)
