import subprocess
import sys

run_attack_parallel_path = '../../textattack/run_attack_parallel.py'

def main():
    datasets = ['mr','yelp-sentiment','imdb']
    models = ['bert']
    recipes = ['tf-adjusted']
    se_thresh_vals = sys.argv[1].split(',')
    for dataset in datasets:
        for model in models:
            for recipe in recipes:
                for se_thresh in se_thresh_vals:
                    args = ['python',run_attack_parallel_path]
                    args.extend(['--model',model+'-'+dataset])
                    args.extend(['--recipe',recipe+':'+se_thresh])
                    args.extend(['--enable_csv','plain'])
                    args.extend(['--num_examples','10'])
                    p = subprocess.Popen(args)
                    if p.wait() != 0:
                        print('Error running attack')
                    break

if __name__ == '__main__':
    main()
