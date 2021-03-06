# JSP-GA-WoC

## Job Shop Scheduling Problem using Genetic Algorithm and Wisdom of Crowds


### Generate Data using generate-jsp-data.py

```
usage: generate-jsp-data.py [-h] [-j JOBS] [-o OPERATIONS] [-m MACHINES] tsp_filepath

FJSP – Test Data Generator

positional arguments:
  tsp_filepath          TSP data file

optional arguments:
  -h, --help            show this help message and exit
  -j JOBS, --jobs JOBS  Number of jobs (default 4)
  -o OPERATIONS, --operations OPERATIONS
                        Maximum number of operations per job (default 4)
  -m MACHINES, --machines MACHINES
                        Number of machines (default 4)
```

### Main usage of fjsp-ga-woc.py

```
usage: fjsp-ga-woc.py [-h] [-g GENERATIONS] [-p POPULATION_SIZE] [-s MATING_SIZE] [-a AI_SIZE] jsp_filepath

Flexible Job Scheduling Problem (FJSP) – Genetic Algorithm w/ Wisdom of Crowds

positional arguments:
  jsp_filepath          JSP data file

optional arguments:
  -h, --help            show this help message and exit
  -g GENERATIONS, --generations GENERATIONS
                        Number of generations to produce (default 100)
  -p POPULATION_SIZE, --population-size POPULATION_SIZE
                        Size of inital population (default 20)
  -s MATING_SIZE, --mating-size MATING_SIZE
                        Size of mating pool (default 10)
  -a AI_SIZE, --ai-size AI_SIZE
                        Size of the AI crowd to poll (default 10)
```
