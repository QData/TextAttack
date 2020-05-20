import colored
import io
import os
import re
import shlex
import signal
import sys
import subprocess
import traceback

def color_text(s, color):
    return colored.stylize(s, colored.fg(color))
    
stderr_file_name = 'err.out'

MAGIC_STRING = '/.*/'
def compare_output_equivalence(desired_output, test_output):
    """ Desired outputs have the magic string '/.*/' inserted wherever the 
        outputat that position doesn't actually matter. (For example, when the 
        time to execute is printed, or another non-deterministic feature of the 
        program.)
        
        `compare_outputs` makes sure all of the outputs match in between
        the magic strings. If they do, it returns True.
    """
    output_pieces = desired_output.split(MAGIC_STRING)
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
        test_output, errored = self.execute()
        if (not errored) and compare_output_equivalence(self.output, test_output):
            self.log_success()
            return True
        else:
            self.log_failure(test_output, errored)
            return False
    
    def log_start(self):
        print(f'Executing test {color_text(self.name, "blue")}.')
        
    def log_success(self):
        success_text = f'✓ Succeeded.'
        print(color_text(success_text, 'green'))
    
    def log_failure(self, test_output, errored):
        fail_text = f'✗ Failed.'
        print(color_text(fail_text, 'red'))
        if errored:
            print(f'Test exited early with error: {test_output}')
        else:
            output1 = f'Test output: {test_output}.'
            output2 = f'Correct output: {self.output}.'
            print(f'\n{output1}\n{output2}\n')

class CommandLineTest(TextAttackTest):
    """ Runs a command-line command to check for desired output. """
    def __init__(self, command, name=None, output=None, desc=None):
        if command is None:
            raise ValueError('Cannot initialize CommandLineTest without command')
        self.command = command
        super().__init__(name=name, output=output, desc=desc)
        
    def execute(self):
        stderr_file = open(stderr_file_name, 'w+')
        if isinstance(self.command, tuple):
            # Support pipes via tuple of commands
            procs = []
            for i in range(len(self.command) - 1):
                if i == 0:
                    proc = subprocess.Popen(shlex.split(self.command[i]), stdout=subprocess.PIPE)
                else:
                    proc = subprocess.Popen(shlex.split(self.command[i]), stdout=subprocess.PIPE, stdin=proc.stdout)
                procs.append(proc)
            # Run last commmand
            result = subprocess.run(
                shlex.split(self.command[-1]), stdin=procs[-1].stdout, 
                stdout=subprocess.PIPE, stderr=stderr_file 
            )
            # Wait for all intermittent processes
            for proc in procs:
                proc.wait()
        else:
            result = subprocess.run(
                shlex.split(self.command), 
                stdout=subprocess.PIPE,
                stderr=stderr_file 
            )
        stderr_file.seek(0) # go back to beginning of file so we can read the whole thing
        stderr_str = stderr_file.read()
        # Remove temp file.
        remove_stderr_file()
        if result.returncode == 0:
            # If the command succeeds, return stdout.
            return result.stdout.decode(), False
        else:
            # If the command returns an exit code, return stderr.
            return stderr_str, True

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
        try:
            with Capturing() as output_lines:
                self.function()
            output = '\n'.join(output_lines)
            return output, False
        except: # catch *all* exceptions
            exc_str_lines = traceback.format_exc().splitlines()
            exc_str = '\n'.join(exc_str_lines)
            return exc_str, True

def remove_stderr_file():
    # Make sure the stderr file is removed on exit.
    try:
        os.unlink(stderr_file_name)
    except FileNotFoundError: 
        # File doesn't exit - that means we never made it or already cleaned it up
        pass
    
def exit_handler(_,__): 
    remove_stderr_file()

# If the program exits early, make sure it didn't create any unneeded files.
signal.signal(signal.SIGINT, exit_handler)