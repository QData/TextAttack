# Minimal makefile for Sphinx documentation
#

# First run "sphinx-apidoc  -f -o apidoc -d 6 -E -T -M ../textattack" 
# Then run "make html"

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# For autobuild
# https://pypi.org/project/sphinx-autobuild/
livehtml:
	sphinx-autobuild -b html $(SPHINXOPTS) "$(BUILDDIR)/html"

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

