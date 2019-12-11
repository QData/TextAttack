import os
import pandas as pd

input_path = '../../textattack/outputs/'
output_path = 'examples.csv'

def main():
    df = pd.DataFrame()
    for run_folder in os.listdir(input_path):
        run_path = os.path.join(input_path, run_folder)
        run_files = os.listdir(run_path)
        if 'final.txt' not in run_files:
            continue
        args_path = os.path.join(run_path,'args.txt')
        args_lines = open(args_path, 'r').readlines()
        se_thresh_idx = args_lines[0].find('tf-adjusted') + len('tf-adjusted:')
        se_thresh = args_lines[0][se_thresh_idx:]
        se_thresh = se_thresh[:se_thresh.index(' ')]
        words = args_lines[0].split()
        dataset = ''
        for word in words:
            if word[:5] == 'bert-':
                dataset = word[5:]
        for f in os.listdir(run_path):
            if f.find('.csv') == -1:
                continue
            add_df = pd.read_csv(os.path.join(run_path,f), index_col=0)
            add_df['SE_Thresh'] = se_thresh
            add_df['dataset'] = dataset
            df = df.append(add_df)
    df.to_csv(output_path)

if __name__ == '__main__':
    main()
