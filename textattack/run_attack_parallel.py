import datetime
import math
import os
import subprocess
import sys
import torch

num_devices = torch.cuda.device_count()
num_data_points = 20
num_data_points_per_device = int(math.ceil(num_data_points / float(num_devices)))

command_line_args = sys.argv[1:] if len(sys.argv) > 1 else []

today = datetime.datetime.now()
folder_name = os.path.join('outputs', 'attack-' + today.strftime('%Y-%m-%d-%H--%H:%M:%S'))
os.makedirs(folder_name)

current_working_dir = os.path.dirname(os.path.abspath(__file__))
run_attack_path = os.path.join(current_working_dir, 'run_attack.py')

arg_file = open(os.path.join(folder_name, 'args.txt'), 'w')
for i in range(num_devices):
    out_file = open(os.path.join(folder_name, f'out-{i}.txt'), 'w')
    new_env = os.environ.copy()
    new_env['CUDA_VISIBLE_DEVICES'] = str(i)
    args = ['python', run_attack_path]
    args.append('--num_examples') 
    args.append(str(num_data_points_per_device))
    args.append('--num_examples_offset')
    args.append(str(num_data_points_per_device * i))
    full_args = args + command_line_args
    out_file.flush()
    p = subprocess.Popen(full_args, env=new_env, stdout=out_file)
    arg_str = ' '.join(full_args)
    print(f'Started process {i} from args:', arg_str)
    arg_file.write(arg_str + '\n')

arg_file.write('Attack started at ')
arg_file.write(today.strftime('%d/%m/%Y at %H:%M:%S'))
arg_file.write('\n')
arg_file.close()

print('Printing results for {} attacks to {}.'.format(num_devices, folder_name))