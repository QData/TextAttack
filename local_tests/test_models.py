import colored
import io
import os
import re
import sys
import subprocess

def color_text(s, color):
    return colored.stylize(s, colored.fg(color))
    
FNULL = open(os.devnull, 'w')

MAGIC_STRING = '/.*/'
def compare_outputs(true_output, test_output):
    """ Desired have the magic string '/.*/' inserted wherever the output
        at that position doesn't actually matter. (For example,
        when the time to execute is printed, or another non-deterministic
        feature of the program.)
        
        `compare_outputs` makes sure all of the outputs match in between
        the magic strings. If they do, it returns True.
    """
    output_pieces = true_output.split(MAGIC_STRING)
    for piece in output_pieces:
        index_in_test = test_output.find(piece)
        if index_in_test < 0:
            return False
        else:
            test_output = test_output[index_in_test + len(piece):]
    return True
    
class TextAttackTest:
    def __init__(self, name=None, output=None, desc=None):
        if name is None:
            raise ValueError('Cannot initialize TextAttackTest without name')
        if output is None:
            raise ValueError('Cannot initialize TextAttackTest without output')
        if desc is None:
            raise ValueError('Cannot initialize TextAttackTest without description')
        self.name = name
        self.output = output
        self.desc = desc
    
    def execute(self):
        """ Executes test and returns test output. To be implemented by
            subclasses.
        """
        raise NotImplementedError()
    
    def __call__(self):
        """ Runs test and prints success or failure. """
        self.log_start()
        test_output = self.execute()
        if compare_outputs(self.output, test_output):
            self.log_success()
            return True
        else:
            self.log_failure(test_output)
            return False
    
    def log_start(self):
        print(f'Executing test {color_text(self.name, "blue")}.')
        
    def log_success(self):
        success_text = f'✓ Succeeded.'
        print(color_text(success_text, 'green'))
    
    def log_failure(self, test_output):
        fail_text = f'✗ Failed.'
        print(color_text(fail_text, 'red'))
        print('\n')
        print(f'Test output: {test_output}.')
        print(f'Correct output: {self.output}.')

class CommandLineTest(TextAttackTest):
    """ Runs a command-line command to check for desired output. """
    def __init__(self, command, name=None, output=None, desc=None):
        if command is None:
            raise ValueError('Cannot initialize CommandLineTest without command')
        self.command = command
        super().__init__(name=name, output=output, desc=desc)
        
    def execute(self):
        result = subprocess.run(
            self.command.split(), 
            stdout=subprocess.PIPE,
            # @TODO: Collect stderr somewhere. In the event of an error, point user to the error file.
            stderr=FNULL 
        )
        return result.stdout.decode()

class Capturing(list):
    """ A context manager that captures standard out during its execution. 
    
    stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

class PythonFunctionTest(TextAttackTest):
    """ Runs a Python function to check for desired output. """
    def __init__(self, function, name=None, output=None, desc=None):
        if function is None:
            raise ValueError('Cannot initialize PythonFunctionTest without function')
        self.function = function
        super().__init__(name=name, output=output, desc=desc)
    
    def execute(self):
        with Capturing() as output_lines:
            self.function()
        output = '\n'.join(output_lines)
        return output
        