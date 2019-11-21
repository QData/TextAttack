from copy import deepcopy
import datetime
import math
import os
import subprocess
import sys
import torch

from textattack.run_attack import get_args

def validate_args(args):
    """ Some arguments from `run_attack` may not be valid to run in parallel.
        Check for them and throw errors here. """
    if args.interactive:
        raise Error('Cannot run attack in parallel with --interactive set.')
    if not args.num_examples:
        raise Error('Cannot run attack with --num_examples set.')

def main():
    input_args = get_args()
    validate_args(input_args)
    
    num_devices = torch.cuda.device_count()
    num_examples_per_device = int(math.ceil(input_args.num_examples / float(num_devices)))
    
    input_args.num_examples = num_examples_per_device
    
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    run_attack_path = os.path.join(current_working_dir, 'run_attack.py')
    
    today = datetime.datetime.now()
    folder_name = os.path.join(current_working_dir, 'outputs', 'attack-' + today.strftime('%Y-%m-%d-%H--%H:%M:%S'))
    os.makedirs(folder_name)
    
    arg_file = open(os.path.join(folder_name, 'args.txt'), 'w')
    for i in range(num_devices):
        # Create outfile for this thread.
        out_file = open(os.path.join(folder_name, f'out-{i}.txt'), 'w')
        # Create unique environment for this thread.
        new_env = os.environ.copy()
        new_env['CUDA_VISIBLE_DEVICES'] = str(i)
        args = ['python', run_attack_path]
        # Change number of examples in argument list.
        command_line_args_list = sys.argv[1:]
        if '--num_examples' in command_line_args_list:
            _x = command_line_args_list.index('--num_examples')
            command_line_args_list[_x+1] = str(num_examples_per_device)
        elif '--n' in command_line_args_list:
            _x = command_line_args_list.index('--n')
            command_line_args_list[__x+1] = str(num_examples_per_device)
        else:
            command_line_args_list.extend(['--n', str(num_examples_per_device)])
        # Change offset in argument list.
        if '--offset' in command_line_args_list:
            _x = command_line_args_list.index('--offset')
            command_line_args_list[_x+1] = str(num_examples_per_device * i)
        elif '--o' in command_line_args_list:
            _x = command_line_args_list.index('--o')
            command_line_args_list[__x+1] = str(num_examples_per_device * i)
        else:
            command_line_args_list.extend(['--o', str(num_examples_per_device*i)])
        # Format and run command.
        full_args = args + command_line_args_list
        out_file.flush()
        p = subprocess.Popen(full_args, env=new_env, stdout=out_file)
        arg_str = ' '.join(full_args)
        print(f'Started process {i} from args:', arg_str)
        arg_file.write(f'Started process {i} from args: ' + arg_str + '\n')
    
    arg_file.write('Attack started at ')
    arg_file.write(today.strftime('%Y-%m-%d at %H:%M:%S'))
    arg_file.write('\n')
    arg_file.close()
    
    print('Printing results for {} attacks to {}.'.format(num_devices, folder_name))

if __name__ == '__main__': main()