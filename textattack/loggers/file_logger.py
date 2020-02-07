import os
import sys
import terminaltables

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
        self.fout.write('-'*45 + ' Result ' + str(examples_completed) + ' ' + '-'*45 + '\n')
        self.fout.write(result.__str__(color_method=color_method))
        self.fout.write('\n')

    def log_rows(self, rows, title, window_id):
        table_rows = [[title, '']] + rows
        table = terminaltables.SingleTable(table_rows)
        self.fout.write(table.table)

    def log_sep(self):
        self.fout.write('-' * 90 + '\n')
        
