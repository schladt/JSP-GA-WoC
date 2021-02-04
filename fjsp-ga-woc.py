"""
CSE 545-050
Project 5: FJSP ‐ GA with Wisdom of Artificial Crowds
Author: Mike Schladt
Python 3.5 or greater required
Python library matplotlib required
"""

import time
import argparse
import random
import json
import plotly.express as px
import pandas as pd
from datetime import datetime
import numpy as np
import sys
import copy
from operator import itemgetter
import statistics

start_time = time.time()

def main():
    # parse command line options
    parser = argparse.ArgumentParser(description='Flexible Job Scheduling Problem (FJSP) – Genetic Algorithm w/ Wisdom of Crowds')
    parser.add_argument('jsp_filepath', type=argparse.FileType("r"), help='JSP data file')
    parser.add_argument('-g', '--generations', type=int, default=100, help="Number of generations to produce (default 100)")
    parser.add_argument('-p', '--population-size', type=int, default=20, help="Size of inital population (default 20)")
    parser.add_argument('-s', '--mating-size', type=int, default=10, help="Size of mating pool (default 10)")
    parser.add_argument('-a', '--ai-size', type=int, default=10, help="Size of the AI crowd to poll (default 10)")
    args = parser.parse_args()


    # parse input file
    cost_matrix = {}
    jobs = {}
    lines = args.jsp_filepath.readlines()

    for line in lines:
        words = [w.strip() for w in line.split(",")]
        # skip first line
        if words[0] == 'job':
            continue

        job = words[0]
        operation = words[1]
        machine = words[2]
        machine_cost = words[3]

        # create new job if not previously added
        if job not in jobs:
            jobs[job] = []
        
        # add operation to job
        if operation not in jobs[job]:
            jobs[job].append(operation)

        # add new operation to cost matrix if not previously added
        if '{0},{1}'.format(job,operation) not in cost_matrix:
            cost_matrix['{0},{1}'.format(job,operation)] = {}

        # update machine cost
        cost_matrix['{0},{1}'.format(job,operation)][machine] = machine_cost

    # initailize matrix hash table
    aggregation_matrix = {}
    makespan_list = []
    for _ in range(args.ai_size):
        chromosome = genetic(jobs, cost_matrix, args.population_size, args.mating_size, args.generations)
        makespan = get_makespan(chromosome)
        makespan_list.append(makespan)
        for job in chromosome:
            for operation in chromosome[job]:
                key = '{0},{1}'.format(job,operation)
                machine = chromosome[job][operation][0]
                # add key and machine if not previously seen
                if key not in aggregation_matrix:
                    aggregation_matrix[key] = {}
                if machine not in aggregation_matrix[key]:
                    aggregation_matrix[key][machine] = [0, 0]

                # update machine count and average makespan
                aggregation_matrix[key][machine][0] += 1
                aggregation_matrix[key][machine][1] = aggregation_matrix[key][machine][1] + makespan

    # create priority list from each task
    priority_list = []
    for key in aggregation_matrix:
        best_heuristic = 0
        for machine in aggregation_matrix[key]:
            avg_makespan = float(aggregation_matrix[key][machine][1]/aggregation_matrix[key][machine][0])
            heuristic = (1/avg_makespan) * aggregation_matrix[key][machine][0]
            if heuristic > best_heuristic:
                best_heuristic = heuristic
                best_machine = machine
        priority_list.append([key, best_machine, best_heuristic])

    # sort priority list by heuristic
    priority_list.sort(key=itemgetter(2), reverse=False)


    # create chromosome shell
    chromosome = {}
    for op_code in priority_list:
        job, operation = op_code[0].split(',')
        if job not in chromosome:
            chromosome[job] = {}
        if operation not in chromosome[job]:
            chromosome[job][operation] = [op_code[1], None, None]
    

    # schedule chromosome using priority list constructed from heuristic
    chromosome = schedule_chromesome(chromosome,cost_matrix,[x[0] for x in priority_list])

    mean = statistics.mean(makespan_list)
    makespan_list.sort()
    low = makespan_list[0]
    high = makespan_list[-1]
    print(low, mean, high, get_makespan(chromosome))
    graph_gantt(chromosome)

def genetic(jobs, cost_matrix, population_size, mating_size, generations):
    """
    INPUT jobs (dict of list)
    INPUT cost_matrix (dict of dict w/ cost values)
    INPUT population_size (int) - size of inital population
    INPUT mating_size (int) - size of mating pool
    INPUT generations (int) - number of 
    OUTPUT best chromosome as determined by GA (dict of dicts w/ machine and start and stop time tuple)
    """

    # generate populations as list of random chromosomes
    population = []
    for _ in range(population_size):
        population.append(gen_chromosome(jobs, cost_matrix))

    progress = [] # stores current best for burndown
    for _ in range (generations):

        for _ in range(population_size):
            population.append(gen_chromosome(jobs, cost_matrix))

        # calculate fitness for population
        makespan_scores = []
        for chromosome in population:
            makespan_scores.append([get_makespan(chromosome),chromosome])
        
        makespan_scores.sort(key=itemgetter(0), reverse=False)

        # survival of the fittest 
        # choose the top chromosomes for the mating pool

        # split mating pool into two groups
        group_size = int(mating_size/2)
        group_a = makespan_scores[:group_size]
        group_b = makespan_scores[group_size:group_size*2]
        
        # print("Best makespan so far: {0}".format(makespan_scores[0][0]))
        progress.append(makespan_scores[0][0])

        # reset population for next generation
        population = []

        # create children with crossover and mutation
        for i in range(len(group_a)):
            # children start as copies of parents
            # *** MUST USE DEEP COPY FOR NESTED LIST ***
            parent_1 = copy.deepcopy(group_a[i][1])
            parent_2 = copy.deepcopy(group_b[i][1])

            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)

            # select job to crossover
            job_index = random.choice(list(child_1.keys()))

            # swap jobs to complete crossover
            temp = child_1[job_index]
            child_1[job_index] = child_2[job_index]
            child_2[job_index] = temp
            
            # mutate each child (the scheduling function is stochastic in nature)    
            mutated_child_1 = copy.deepcopy(child_1)
            mutated_child_2 = copy.deepcopy(child_2)
            mutated_child_1 = schedule_chromesome(mutated_child_1, cost_matrix)
            mutated_child_2 = schedule_chromesome(mutated_child_2, cost_matrix)

            # add to all children to population
            # fittest will be selected during next round

            population.append(parent_1)
            population.append(parent_2)
            population.append(child_1)
            population.append(child_2)
            population.append(mutated_child_1)
            population.append(mutated_child_2)

    # calculate final makespan for population
    makespan_scores = []
    for chromosome in population:
        makespan_scores.append([get_makespan(chromosome),chromosome])
    
    print("Best makespan (final): {0}".format(makespan_scores[0][0]))
    progress.append(makespan_scores[0][0])

    # graph progess
    # fig = px.line(y=progress, x=list(range(len(progress))))
    # fig.update_layout(
    #     xaxis_title="Generation Number",
    #     yaxis_title="Makespan Time",
    # )
    # name = 'figures\\{0}.html'.format(datetime.now()).replace(':','').replace(' ','_')
    # fig.write_html((name), auto_open=True)

    # return best chromosome
    return makespan_scores[0][1]

def gen_chromosome(jobs, cost_matrix):
    """
    Generates a pseudo-random chromosome
    INPUT jobs (dict of list)
    INPUT cost_matrix (dict of dict w/ cost values)
    OUTPUT chromosome (dict of dicts w/ machine and start time tuple)
    """

    chromosome = {}

    # machines can be derived by looking at the first job and operation of the cost matrix
    job_keys = list(jobs.keys())
    j1 = job_keys[0]
    o1 = jobs[j1][0]
    machines = list(cost_matrix['{0},{1}'.format(j1,o1)].keys())

    # randomly assign machines to each operation
    for job in jobs:
        # add job to chromosome
        if job not in chromosome:
            chromosome[job] = {}
        for operation in jobs[job]:
            # create machine assignment
            machine_assignment = random.choice(machines)
            # start and stop time set to null - to be filled by schedule_chromosome 
            chromosome[job][operation] = ['{0}'.format(machine_assignment), None, None]

    chromosome = schedule_chromesome(chromosome, cost_matrix)

    return chromosome

def schedule_chromesome(chromosome, cost_matrix, priority_list=None):
    """
    Fills in machine start times for given chromosome
    INPUT: chromosome (dict of dicts)
    INPUT: INPUT cost_matrix (dict of dict w/ cost values)
    OUTPUT chromosome (dict of dicts w/ machine and start and stop time tuple)
    """        

    # generate priority list if not provided
    if priority_list is None:
        # find total number of operations to schedule
        num_operations = 0
        for job in chromosome:
            for operation in chromosome[job]:
                num_operations += 1
    
        # randomly determine scheduling priority 
        priority_list = []

        while len(priority_list) < num_operations:
            job = random.choice(list(chromosome.keys()))
            for operation in chromosome[job]:
                op_name = '{0},{1}'.format(job, operation)
                if op_name not in priority_list:
                    priority_list.append(op_name)
                    break

    # fill in machine start times of the chromosome
    machine_times = {}
    for op_pair in priority_list:
        words = op_pair.split(',')
        job = words[0]
        operation = words[1]

        # get the assignment machine from the chromosome
        machine = chromosome[job][operation][0]
        cost = cost_matrix[op_pair][machine]
        
        # add machine to machine time tracker
        if machine not in machine_times:
            machine_times[machine] = 0
        
        # find start and finsih times
        # find previous operation
        prev_op_finish = 0
        op_keys = list(chromosome[job].keys())
        index = op_keys.index(operation)
        if index != 0:
            prev_op_finish = chromosome[job][op_keys[index - 1]][2]

        # start time will be the later of machine ready time and previous operation finish
        start = max(machine_times[machine], prev_op_finish)
        finish = start + int(cost)

        # set start and finish time to current machine time
        chromosome[job][operation][1] = start
        chromosome[job][operation][2] = finish

        # add cost to current machine time
        machine_times[machine] = finish

    return chromosome

def get_makespan(chromosome):
    """
    Helper function to find make span
    INPUT chromosome (dict of dicts w/ machine and start and stop time tuple)
    OUTPUT makespan - int
    """

    last_stop = 0
    for job in chromosome:
        for operation in chromosome[job]:
            stop_time = chromosome[job][operation][2]
            if stop_time > last_stop:
                last_stop = stop_time

    return last_stop

def graph_gantt(chromosome):
    """
    INPUT chromosome (dict of dicts w/ machine and start and stop time tuple)
    OUTPUT HTML doc with Gantt Chart 
    """

    def convert_to_datetime(x):
        return '{0}'.format(datetime.fromtimestamp(31556926+x))
        # return datetime.fromtimestamp(31556926+x*24*3600).strftime("%Y-%d-%m")

    data = []
    for job in chromosome:
        for operation in chromosome[job]:
            task = '{0},{1}'.format(job,operation)
            machine = chromosome[job][operation][0]
            start = chromosome[job][operation][1]
            finish = chromosome[job][operation][2]
            data.append(dict(Task=task, Start=convert_to_datetime(start), Finish=convert_to_datetime(finish), Machine=machine, Job=job))

    makespan = get_makespan(chromosome)

    df = pd.DataFrame(data)

    num_tick_labels = np.linspace(start = 0, stop = makespan, num = makespan+1, dtype = int)
    date_ticks = [convert_to_datetime(x) for x in num_tick_labels]

    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Machine", color="Job")
    fig.layout.xaxis.update({
            'tickvals' : date_ticks,
            'ticktext' : num_tick_labels
            })
    name = 'figures\\{0}.html'.format(datetime.now()).replace(':','').replace(' ','_')
    fig.write_html((name), auto_open=True)


if __name__ == '__main__':
    main()

print("Executed in {0} seconds".format(time.time() - start_time))

