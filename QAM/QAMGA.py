# Author: Minh Hua
# Date: 12/5/2022
# Purpose: This module contains the main evolutionary algorithm that will be used to evolve QAM constellation diagrams.

import random
import numpy as np
import math
import copy
import os

from constellation import Constellation

class GACE:
    """Genetic Algorithms for Constellation Evolution evolves a population of QAM constellations."""
    def __init__(self,
        n:int,
        eval_metric:str='Gray',
        pop_size:int=1000,
        crossover_rate:float=0.8,
        mutation_rate:float=0.2,
        init_strat:str='random',
        sel_strat:str='tournament',
        tournament_sel_k:int=5,
        crossover_strat:str='one-pt',
        mutation_strat:str='uniform',
        replace_strat:str='replace-all-parents',
        top_k_elitism:int=None,
        max_fitness_evals:int=10000,
        early_stop:bool=False,
        early_stop_gen:int=5,
        early_stop_thresh:float=10**-4,
        print_debug:bool=False,
        save_output_path:str=None,
        save_every:int=None
    ) -> None:
        """
        Description:
            Initializes an instance of the Genetic Algorithms for Constellation Evolution model.

        Arguments:
            n: the number of bits.
            eval_metric: the metric to determine the fitness of individuals.
                'Gray': the Gray score of the constellation.
            pop_size: the size of the population.
            crossover_rate: the crossover rate.
            mutation_rate: the mutation rate.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            sel_strat: the crossover strategy.
                'tournament': tournament selection.
            tournament_sel_k: the number of individuals to include in a tournament.
            crossover_strat: the crossover strategy.
                'one-pt': one-point crossover.
            mutation_strat: the mutation strategy.
                'uniform': uniform mutation.
            replace_strat: the population replacement strategy.
                'replace-all-parents':
            top_k_elitism: top-k elitism.
            max_fitness_evals: the maximum number of fitness evaluations.
            early_stop: whether or not to stop evolving early.
            early_stop_gen: the number of generations to check for fitness improvement before stopping early.
            early_stop_thresh: the threshold to check whether the best fitness has changed.
            print_debug: whether or not to print debug information.
            save_output_path: the path to save the results
            save_every: save output every mapping that gets processed.

        Return:
            (None)
        """
        # general parameters
        self.n = n

        # evolutionary algorithm parameters
        self.eval_metric = eval_metric
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.init_strat = init_strat
        self.sel_strat = sel_strat
        self.tournament_sel_k = tournament_sel_k
        self.crossover_strat = crossover_strat
        self.mutation_strat = mutation_strat
        self.replace_strat = replace_strat
        self.top_k_elitism = top_k_elitism
        self.max_fitness_evals = max_fitness_evals
        self.early_stop = early_stop
        self.early_stop_gen = early_stop_gen
        self.early_stop_thresh = early_stop_thresh

        # data saving parameters
        self.print_debug = print_debug
        self.save_output_path = save_output_path
        self.save_every = save_every

        # count the number of fitness evaluations
        self.fitness_evaluations = 0

    def init_pop(self) -> list:
        """
        Description:
            Initializes a population of constellation diagram individuals.

        Arguments:
            strategy: the strategy used to initialize the population.
                'random': uniformly initialize the individual chromosomes.

        Return:
            (list) the list of initial IntertaskMapping individuals.
        """
        population = []
        if self.init_strat == 'random': # randomly generate mappings. Each mapping is sampled uniformly
            for _ in range(self.pop_size):
                constellation_individual = Constellation(self.n)
                constellation_individual.constellation, constellation_individual.zeros_locs, constellation_individual.ones_locs = constellation_individual.initialize_constellation()
                constellation_individual.ID = constellation_individual.create_ID()
                constellation_individual.fitness = self.evaluate_fitness(constellation_individual)
                # append individual to the population
                population.append(constellation_individual)
        # return initial population
        return population

    def select_parents(self) -> Constellation:
        """
        Description:
            Select parents for crossover according to some strategy.

        Arguments:
            strategy: the strategy used to crossover the parents.
                'tournament': tournament selection.
                'fitness-proportionate': fitness-proportionate selection.

        Return:
            (Constellation) a single Constellation parent chosen from a selection method.
        """
        if self.sel_strat == 'tournament':
            parents = random.choices(self.population, k = self.tournament_sel_k)
            parent = sorted(parents, key = lambda agent: agent.fitness, reverse=True)
            parent = parent[0]
        elif self.sel_strat == 'fitness-proportionate':
            r = random.random() * sum(mapping.fitness for mapping in self.population)
            x = 0
            for mapping in self.population:
                x += mapping.fitness
                if r <= x:
                    parent = mapping
                    break
        return parent # return parent

    def check_01s(self, rail) -> tuple:
        zeros = 0
        ones = 0
        for x in range(rail):
            for y in range(rail[x]):
                if rail[x][y] == 0:
                    zeros += 1
                elif rail[x][y] == 1:
                    ones += 1
        return zeros, ones

    def repair(self, rail:np.array, zeros:int, ones:int) -> np.array:
        if zeros > ones:
            to_fill = 0
            num_to_fill = 
        elif ones < zeros:
            to_fill = 1
        else:
            return rail

    def crossover(self, parent_1:Constellation, parent_2:Constellation) -> list:
        """
        Description:
            Generate a number of offspring using certain crossover strategies.
            May result in infeasible rails.

        Arguments:
            parent_1: the first parent.
            parent_2: the second parent.
            strategy: the strategy used to crossover the parents.
                'one-pt': one-point crossover.
                'two-pt': two-point crossover.
                'fusion': select the bit from the higher-fitness parent.

        Return:
            (list) a list of Constellation offspring.
        """
        offspring = []
        if self.crossover_strat == 'one-pt':
            # randomly select a point to crossover the parents in each rail
            offspring_1 = Constellation(self.n)
            offspring_2 = Constellation(self.n)
            x_crossover_pt = np.random.choice(len(parent_1.constellation), size = 1)[0]
            y_crossover_pt = np.random.choice(len(parent_1.constellation[0]), size = 1)[0]
            # copy rails from parents to child
            for rail_idx in range(self.n):
                offspring_1_rail = np.ones((parent_1.config_dim, parent_1.config_dim), dtype=np.int8)
                offspring_2_rail = np.ones((parent_1.config_dim, parent_1.config_dim), dtype=np.int8)
                if not x_crossover_pt == 0:
                    offspring_1_rail[:x_crossover_pt] = parent_1.constellation[rail_idx][:x_crossover_pt]                   
                offspring_1_rail[x_crossover_pt][:y_crossover_pt] = parent_1.constellation[rail_idx][x_crossover_pt][:y_crossover_pt]
                offspring_1_rail[x_crossover_pt][y_crossover_pt:] = parent_2.constellation[rail_idx][x_crossover_pt][y_crossover_pt:]          
                offspring_1_rail[x_crossover_pt + 1:] = parent_2.constellation[rail_idx][x_crossover_pt + 1:]

                if not x_crossover_pt == 0:
                    offspring_2_rail[:x_crossover_pt] = parent_2.constellation[rail_idx][:x_crossover_pt]                   
                offspring_2_rail[x_crossover_pt][:y_crossover_pt] = parent_2.constellation[rail_idx][x_crossover_pt][:y_crossover_pt]
                offspring_2_rail[x_crossover_pt][y_crossover_pt:] = parent_1.constellation[rail_idx][x_crossover_pt][y_crossover_pt:]          
                offspring_2_rail[x_crossover_pt + 1:] = parent_1.constellation[rail_idx][x_crossover_pt + 1:]

                offspring_1.constellation.append(offspring_1_rail)
                offspring_1.zeros_locs.append([])
                offspring_1.ones_locs.append([])
                offspring_1.zeros_locs[-1], offspring_1.ones_locs[-1] = offspring_1.record_01_locs(offspring_1_rail, parent_1.empty_coords)

                offspring_2.constellation.append(offspring_2_rail)
                offspring_2.zeros_locs.append([])
                offspring_2.ones_locs.append([])
                offspring_2.zeros_locs[-1], offspring_2.ones_locs[-1] = offspring_2.record_01_locs(offspring_2_rail, parent_1.empty_coords)                
                
            # append the offspring 
            offspring_1.ID = offspring_1.create_ID()
            offspring_1.fitness = self.evaluate_fitness(offspring_1)
            offspring_1.empty_coords = parent_1.empty_coords
            offspring_2.ID = offspring_2.create_ID()
            offspring_2.fitness = self.evaluate_fitness(offspring_2)
            offspring_2.empty_coords = parent_1.empty_coords
            offspring.append(offspring_1)
            offspring.append(offspring_2)
        elif self.crossover_strat == 'two-pt':
            pass
        elif self.crossover_strat == 'fusion':
            pass
        # return the children
        return offspring

    def mutate(self, individual:Constellation) -> Constellation:
        """
        Description:
           Mutate a Constellation individual.

        Arguments:
            individual: the Constellation individual to mutate.
            strategy: the strategy used to mutate the individual.
                'uniform': mutate each gene with equal probability.
                'weighted': 

        Return:
            (Constellation) the mutated Constellation individual.
        """
        if self.mutation_strat == 'uniform':
            individual.random_swap()
            individual.ID = individual.create_ID()
            individual.fitness = self.evaluate_fitness(individual)   
        # return the mutated individual
        return individual

    def replace(self, offspring:list) -> list:
        """
        Description:
            Replace the current population with the new offspring population using different strategies.

        Arguments:
            offspring: a list of the current offspring.
            strategy: the replacement strategy.
                'replace-all-parents': canonical GA replacement strategy where the offspring replaces the parent population.

        Return:
            (list) a list of the new population.
        """
        if self.replace_strat == 'replace-all-parents':
            return offspring

    def evaluate_fitness(self, constellation_ind:Constellation, set_fitness:float=True) -> float:
        """
        Description:
            Evaluate the fitness of the mapping.

        Arguments:
            constellation_ind: the Constellation to evaluate the fitness for.
            set_fitness: wether or not we want to set the individual's fitness after evaluation.

        Return:
            (float) the fitness of the Constellation individual.
        """
        # count the number of fitness evaluations
        self.fitness_evaluations = self.fitness_evaluations + 1
        composed_constellation = constellation_ind.compose_config(constellation_ind.constellation)
        fitness = abs(1 - constellation_ind.calc_gray_score(composed_constellation))
        if set_fitness:
            constellation_ind.fitness = fitness
        return fitness

    def determine_best_fit(self, population:list) -> Constellation:
        """
        Description:
            Determine the individual with the best fitness in the population.

        Arguments:
            population: the current population to evaluate the fitness for.

        Return:
            (Constellation) the most fit individual.
        """
        return sorted(population, key = lambda agent: agent.fitness, reverse=True)[0]        

    def evolve(self) -> list:
        """
        Description:
            Run the GACE model and generate a list of constellations.

        Arguments:
            

        Return:
            (list) a list of the most fit constellations.
        """
        try:
            # if we save data
            if self.save_output_path:
                with open(self.save_output_path, 'w') as f:
                    f.write("Results\n")
                str_builder = ""

            # initialize initial population
            self.population = self.init_pop()
            # evaluate initial population's fitness
            for constellation in self.population:
                # print debug info
                if self.print_debug:
                    print('Initial Indivial ID: {}, Fitness: {}'.format(constellation.ID, constellation.fitness))
                    if self.save_output_path:
                        str_builder += 'Initial Indivial ID: {}, Fitness: {}\n'.format(constellation.ID, constellation.fitness)

            # if we want to stop early, we have to keep track of the best fitness
            if self.early_stop:
                best_fitness = [self.determine_best_fit(self.population)]

            gen = 0

            # main evolution loop
            while self.fitness_evaluations < self.max_fitness_evals:
                if self.print_debug:
                    print("Generation {}".format(gen))

                # elitism
                if self.top_k_elitism:
                    # save top k individuals from parent population
                    top_k = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[:self.top_k_elitism]
                        
                offspring = []
                # generate a number of offspring
                while len(offspring) < self.pop_size:
                    # select parents for crossover
                    parent_1 = self.select_parents()
                    parent_2 = self.select_parents()
                    # make sure we do not have identical parents
                    while parent_2.ID == parent_1.ID:
                        parent_2 = self.select_parents()
                    if random.random() < self.crossover_rate:
                        # generate offspring using crossover
                        new_offspring = self.crossover(parent_1, parent_2)
                    else: # offspring are exact copies of the parents
                        new_offspring = [parent_1, parent_2]
                    # mutate offspring
                    for offspring_idx in range(len(new_offspring)):
                        new_offspring[offspring_idx] = self.mutate(new_offspring[offspring_idx])
                    # evaluate offspring fitness
                    for offspring_soln in new_offspring:
                        # add offspring to temporary offspring array
                        offspring.append(offspring_soln)
                    # print debug info
                    if self.print_debug:
                        for offspring_soln in new_offspring:
                            print('Offspring ID: {}, Fitness: {}'.format(offspring_soln.ID, offspring_soln.fitness))
                    if self.save_output_path:
                        for offspring_soln in new_offspring:                    
                            str_builder += 'Offspring ID: {}, Fitness: {}\n'.format(offspring_soln.ID, offspring_soln.fitness)                   

                # replace population with offspring
                self.population = self.replace(offspring)

                # elitism
                if self.top_k_elitism:
                    self.population = sorted(self.population + top_k, key=lambda agent: agent.fitness, reverse=True)[:self.pop_size]                 

                # save info, print info, analyze search
                if self.save_every and gen % self.save_every == 0:
                    with open(self.save_output_path, 'a') as f:
                        f.write(str_builder)
                        str_builder = ""

                # determine early stop if needed
                if self.early_stop:
                    best_fitness.append(self.determine_best_fit(self.population))
                    if len(best_fitness) >= self.early_stop_gen:
                        # check to see if the best fitness has changed enough from the average of the past window
                        moving_average = sum(f.fitness for f in best_fitness[-self.early_stop_gen:]) / self.early_stop_gen
                        if self.print_debug:
                            print("Average best fitness of the past {} generations: {}".format(self.early_stop_gen, moving_average))
                        if abs(best_fitness[-1].fitness - moving_average) <= self.early_stop_thresh:
                            # we can stop early
                            if self.print_debug:
                                print("Stopping early.")
                            break
                
                gen += 1

            if self.save_output_path:
                str_builder = "Final population:\n"
                for ind in self.population:
                    str_builder += 'Offspring ID: {}, Fitness: {}\n'.format(ind.ID, ind.fitness)
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)

        except KeyboardInterrupt:
            if self.save_output_path:
                with open(self.save_output_path, 'a') as f:
                    f.write(str_builder)

if __name__ == "__main__":
    # general parameters
    n = 4

    # evolutionary algorithm parameters
    eval_metric = "Gray"
    pop_size = 10
    crossover_rate = 0.8
    mutation_rate = 0.05
    init_strat = 'random'
    sel_strat = 'tournament'
    tournament_sel_k = int(0.1 * pop_size)
    crossover_strat = 'one-pt'
    mutation_strat = 'uniform'
    replace_strat = 'replace-all-parents'
    top_k_elitism = None
    max_fitness_evals = 2000
    early_stop = False
    early_stop_gen = None
    early_stop_thresh = None

    # data saving parameters
    print_debug = True
    save_output_path = None
    save_every = None

    ea = GACE(n, eval_metric, pop_size, crossover_rate, mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat,
    mutation_strat, replace_strat, top_k_elitism, max_fitness_evals, early_stop, early_stop_gen, early_stop_thresh, print_debug,
    save_output_path, save_every)

    population = ea.evolve()

    # print the final evolved population
    for mapping in ea.population:
        print(mapping)