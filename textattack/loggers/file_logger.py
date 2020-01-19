import sys
import os

from .logger import Logger

class FileLogger(Logger):
    def __init__(self, filename='', stdout=False):
        self.stdout = stdout
        if stdout:
            self.fout = sys.stdout
        elif isinstance(filename, str):
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.fout = open(filename, 'w')
        else:
            self.fout = filename

    def log_attack_result(self, result, examples_completed):
        color_method = 'stdout' if self.stdout else 'file'
        self.fout.write('-'*35 + ' Result ' + str(examples_completed) + ' ' + '-'*35 + '\n')
        self.fout.write(result.__str__(color_method=color_method))
        self.fout.write('\n')

    def log_rows(self, rows, title, window_id):
        for row in rows:
            self.fout.write(f'{row[0]} {row[1]}\n')

    def log_sep(self):
        self.fout.write('-'*80 + '\n')
        
