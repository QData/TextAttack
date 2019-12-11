import os
import subprocess
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
run_attack_parallel_path = os.path.join(dir_path, '../../textattack/run_attack_parallel.py')

def main():
    datasets = ['imdb']
    models = ['lstm']
    recipes = ['alz-adjusted']
    se_thresh_vals = sys.argv[1].split(',')
    for dataset in datasets:
        for model in models:
            for recipe in recipes:
                for se_thresh in se_thresh_vals:
                    args = ['python', '-u', run_attack_parallel_path]
                    args.extend(['--model',model+'-'+dataset])
                    args.extend(['--recipe',recipe+':'+se_thresh])
                    args.extend(['--enable_csv','plain'])
                    args.extend(['--num_examples','1000'])
                    p = subprocess.Popen(args)
                    if p.wait() != 0:
                        print('Error running attack')

if __name__ == '__main__':
    main()
