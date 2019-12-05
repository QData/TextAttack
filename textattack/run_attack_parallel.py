from copy import deepcopy
import datetime
import math
import os
import re
import subprocess
import sys
import torch

from textattack.run_attack import get_args
from textattack.utils import color


def _cb(s): return color(str(s), color='blue', method='stdout')
def _cg(s): return color(str(s), color='green', method='stdout')
def _cr(s): return color(str(s), color='red', method='stdout')

result_regex = '----------------------------------- Result [0-9]* -----------------------------------'
arg_eq_regex = '^--[A-Za-z0-9_-]*\=[A-Za-z0-9_-]*$'

def validate_args(args):
    """ Some arguments from `run_attack` may not be valid to run in parallel.
        Check for them and throw errors here. """
    if args.interactive:
        raise Error('Cannot run attack in parallel with --interactive set.')
    if not args.num_examples:
        raise Error('Cannot run attack with --num_examples set.')

def get_command_line_args():
    """ Splits args like `--arg=4` into `--arg 4`. We can't use the argparser
        for this because we need to parse the args back into a string to pass to
        run_attack. """
    raw_args = sys.argv[1:]
    args = []
    for arg in raw_args:
        if re.match(arg_eq_regex, arg):
            args.extend(arg.split('='))
        else:
            args.append(arg)
    return args

def main():
    input_args = get_args()
    validate_args(input_args)
    
    num_devices = torch.cuda.device_count()
    num_examples_per_device = int(math.ceil(input_args.num_examples / float(num_devices)))
    
    input_args.num_examples = num_examples_per_device
    
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    run_attack_path = os.path.join(current_working_dir, 'run_attack.py')
    
    today = datetime.datetime.now()
    out_dir = input_args.out_dir or 'outputs'
    folder_name = os.path.join(current_working_dir, out_dir, 'attack-' + today.strftime('%Y-%m-%d--%H:%M:%S'))
    os.makedirs(folder_name)
    
    arg_file = open(os.path.join(folder_name, 'args.txt'), 'w')
    processes = []
    out_file_names = []
    for i in range(num_devices):
        # Create outfiles for this thread.
        out_file_name = os.path.join(folder_name, f'out-{i}.txt')
        out_file = open(out_file_name, 'w')
        out_file_names.append(out_file_name)
        err_file = open(os.path.join(folder_name, f'err-{i}.txt'), 'w')
        # Create unique environment for this thread.
        new_env = os.environ.copy()
        new_env['CUDA_VISIBLE_DEVICES'] = str(i)
        new_env['PYTHONUNBUFFERED'] = '1'
        args = ['python', run_attack_path]
        command_line_args_list = get_command_line_args()
        # Change number of examples in argument list.
        examples_at_i = str(num_examples_per_device)
        if '--num_examples' in command_line_args_list:
            _x = command_line_args_list.index('--num_examples')
            command_line_args_list[_x+1] = examples_at_i
        else:
            command_line_args_list.extend(['--num_examples', examples_at_i])
        # Change offset in argument list.
        offset_at_i = str(input_args.num_examples_offset + num_examples_per_device * i)
        if '--num_examples_offset' in command_line_args_list:
            _x = command_line_args_list.index('--num_examples_offset')
            command_line_args_list[_x+1] = offset_at_i
        else:
            command_line_args_list.extend(['--num_examples_offset', offset_at_i])
        
        # Format and run command.
        full_args = args + command_line_args_list
        out_file.flush()
        p = subprocess.Popen(full_args, env=new_env, stdout=out_file, stderr=err_file)
        processes.append(p)
        arg_str = ' '.join(full_args)
        print(f'Started process {i}:', _cr(arg_str), '\n')
        arg_file.write(f'Started process {i}: ' + arg_str + '\n')
    
    arg_file.write('Attack started at ')
    arg_file.write(today.strftime('%Y-%m-%d at %H:%M:%S'))
    arg_file.write('\n')
    arg_file.close()
    
    print('Printing results for {} attack threads to folder {}'.format(_cg(num_devices), 
        _cb(folder_name)))
    
    # Wait for attacks to run and aggregate results.
    for p in processes:
        if p.wait() != 0:
            print('Error running process ', p)
    final_out_file = open(os.path.join(folder_name, 'final.txt'), 'w')
    i = 1
    all_samples = []
    all_accs = []
    all_pert_percs = []
    all_queries = []
    for out_file in out_file_names:
        lines = open(out_file, 'r').readlines()
        j = 0
        samples = 0
        acc = 0
        pertperc = 0
        queries = 0
        while j < len(lines):
            line = lines[j].strip()
            if line.startswith('Number of successful attacks:'):
                samples = int(line.split()[-1])
            if line.startswith('Accuracy under attack:'):
                acc = float(line.split()[-1][:-1])
            if line.startswith('Average perturbed word %:'):
                pertperc = float(line.split()[-1][:-1])
            if line.startswith('Avg num queries:'):
                queries = float(line.split()[-1])
            if re.match(result_regex, lines[j]):
                line_j_tokens = lines[j].split()
                line_j_tokens[2] = str(i)
                lines[j] = ' '.join(line_j_tokens) + '\n'
                i += 1
                if 'FAILED' in lines[j+1]:
                    n=3
                else:
                    n=4
                for _ in range(n):
                    final_out_file.write(lines[j])
                    j += 1
            else: j += 1
        
        all_samples.append(samples)
        all_accs.append(acc)
        all_pert_percs.append(pertperc)
        all_queries.append(queries)


    avg_acc = 0
    avg_pert_perc = 0
    avg_queries = 0
    total_samples = 0
    for num_samples, acc, pert_perc, queries in zip(all_samples, all_accs, all_pert_percs, all_queries):
        total_samples += num_samples
        avg_acc += (acc * num_samples)
        avg_pert_perc += (pert_perc * num_samples)
        avg_queries += (queries * num_samples)

    avg_acc /= float(total_samples)
    avg_pert_perc /= float(total_samples)
    avg_queries /= float(total_samples)

    final_out_file.write('Number of successful attacks: ' + str(total_samples) + '\n')
    final_out_file.write('Accuracy under attack: ' + str(avg_acc) + '\n')
    final_out_file.write('Average perturbed word %: ' + str(avg_pert_perc) + '\n')
    final_out_file.write('Avg num queries: ' + str(avg_queries))
    final_out_file.close()

if __name__ == '__main__': main()
