"""
CSE 545-050
Project 6: FJSP ‐ GA with Wisdom of Artificial Crowds
Author: Mike Schladt
Helper program to generate test data for FJSP
"""

import argparse
import random

parser = argparse.ArgumentParser(description='FJSP – Test Data Generator')
parser.add_argument('tsp_filepath', type=argparse.FileType('w'), help='TSP data file')
parser.add_argument('-j', '--jobs', type=int, default=4, help="Number of jobs (default 4)")
parser.add_argument('-o', '--operations', type=int, default=4, help="Maximum number of operations per job (default 4)")
parser.add_argument('-m', '--machines', type=int, default=4, help="Number of machines (default 4)")
args = parser.parse_args()

args.tsp_filepath.write("job, operation, machine, cost\n")
for i in range(1, args.jobs + 1):
    num_operations = int(args.operations * random.random()) + 1
    for j in range(1, num_operations + 1):
        for k in range(1, args.machines + 1):
            cost = int(10 * random.random()) + 1
            args.tsp_filepath.write("J{0}, O{1}, M{2}, {3}\n".format(i,j,k,cost))
