#! /usr/bin/env python

import sys
import subprocess
import subprocess

def run_shell_file(file_path):
    try:
        subprocess.run(['bash', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    # file_path = sys.argv[1]
    file_path = r"D:\GoogleDrive\01PhD\09Project\synthetic_dataset\reproduce\single_split.sh"
    run_shell_file(file_path)