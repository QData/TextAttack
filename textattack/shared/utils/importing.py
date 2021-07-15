# Code copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py

import importlib
import time
import types

import textattack

from .install import logger


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


def load_module_from_file(file_path):
    """Uses ``importlib`` to dynamically open a file and load an object from
    it."""
    temp_module_name = f"temp_{time.time()}"
    colored_file_path = textattack.shared.utils.color_text(
        file_path, color="blue", method="ansi"
    )
    logger.info(f"Loading module from `{colored_file_path}`.")
    spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
