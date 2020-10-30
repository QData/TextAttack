# Code copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py

import importlib
import types


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    This allows them to only be loaded when they are used.
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Lazy module loader cannot find module named `{self.__name__}`. "
                f"This might be because TextAttack does not automatically install some optional dependencies. "
                f"Please run `pip install {self.__name__}` to install the package."
            ) from e
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
