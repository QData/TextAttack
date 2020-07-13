import shlex
import subprocess


def run_command_and_get_result(command):
    """Runs a command in the console and gets the result.

    Command can be a string (single command) or a tuple of strings
    (multiple commands). In the multi-command setting, commands will be
    joined together with a pipe, and the output of the last command will
    be returned.
    """
    from subprocess import PIPE

    # run command
    if isinstance(command, tuple):
        # Support pipes via tuple of commands
        procs = []
        for i in range(len(command) - 1):
            if i == 0:
                proc = subprocess.Popen(shlex.split(command[i]), stdout=PIPE)
            else:
                proc = subprocess.Popen(
                    shlex.split(command[i]),
                    stdout=subprocess.PIPE,
                    stdin=procs[-1].stdout,
                )
            procs.append(proc)
        # Run last commmand
        result = subprocess.run(
            shlex.split(command[-1]), stdin=procs[-1].stdout, stdout=PIPE, stderr=PIPE
        )
        # Wait for all intermittent processes
        for proc in procs:
            proc.wait()
    else:
        result = subprocess.run(shlex.split(command), stdout=PIPE, stderr=PIPE)
    return result
