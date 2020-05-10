#!/usr/bin/env python

"""
The TextAttack main module:

    A command line parser to run an attack from user specifications.
"""



from textattack.shared.scripts.run_attack_args_helper import get_args
from textattack.shared.scripts.run_attack_parallel import run as run_parallel
from textattack.shared.scripts.run_attack_single_threaded import run as run_single_threaded

if __name__ == '__main__':
    args = get_args()
    if args.parallel:
        run_parallel(args)
    else:
        run_single_threaded(args)
