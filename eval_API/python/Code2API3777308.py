import subprocess

def run_shell_script(script_path):
    return subprocess.call(['sh', script_path])
