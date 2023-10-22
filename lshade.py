import numpy as np
import scipy.stats
from data import Data
import typing
import random
from copy import deepcopy
from util import init_T, is_within_FoV

def keep_bounds(population, bounds):
    return np.clip(population, bounds[:, 0], bounds[:, 1])

def parent_choice(population, n_parents, n_rows):
    pop_size = population.shape[0]
    choices = np.indices((pop_size, pop_size))[1]
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    choices = choices[mask].reshape(pop_size, pop_size-1)
    parents = np.array([np.random.choice(row, n_parents, replace=False) for row in choices[:n_rows]])

    return parents

def apply_mutation(population, archive, population_fitness, f, p, bounds):
    # Return if population size too small
    if len(population) < 4:
        return population
    
    # 1. Find best parents
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))

    if archive:
        union = np.concatenate((population, archive), axis=0)
    else:
        union = population
    
    p_best = np.array(p_best)
    # 2. Choose 2 random parents
    parents = parent_choice(union, 2, population.shape[0])
    mutated = population + f*(population[p_best] - population)
    mutated += f*(union[parents[:, 0]] - union[parents[:, 1]])

    return keep_bounds(mutated, bounds)

def apply_crossover(population, mutated, cr):
    chosen = np.random.rand(*population.shape)
    j_rand = np.random.randint(0, population.shape[1])
    chosen[j_rand::population.shape[1]] = 0
    return np.where(chosen <= cr, mutated, population)

def select(population, new_population, fitness, new_fitness, return_indexes=False):
    indexes = np.where(fitness > new_fitness)[0]
    population[indexes] = new_population[indexes]
    if return_indexes:
        return population, indexes
    else:
        return population

class LSHADEConfig(typing.NamedTuple):
    population_size: int
    memory_size: int
    max_evals: int
    bounds: np.ndarray
    seed: int
    useless_penalty: float
    activated_penalty: float

class LSHADE:
    def __init__(self):
        self._adapted = False
    
    def adapt(self, data: Data):
        self.n = data.n
        self.m = data.m
        self.q = data.q
        self.K = data.K
        self.T = init_T(data)
        # None value atribute
        self.population_size = None

        self._adapted = True
    
    def init_population(self, population_size, bounds):
        population = np.random.uniform(-0.5, self.q + 0.5, size=(population_size, self.n))
        return keep_bounds(population, bounds)
    
    def apply_fitness(self, population, useless_penalty, activated_penalty):
        normalized_population = np.round(population).astype(np.int32)
        normalized_population[normalized_population == -1] = 0
        normalized_population[normalized_population == self.q+1] = self.q
        normalized_population = np.abs(normalized_population)
        f = np.zeros((population.shape[0], self.m), dtype=np.int32)
        useless = np.zeros((population.shape[0], ), dtype=np.int32)
        activated = np.zeros((population.shape[0], ), dtype=np.int32)

        for i, individual in enumerate(normalized_population):
            pan, freq = np.unique(individual, return_counts=True)
            activated[i] = self.n
            if pan[-1] == self.q:
                activated[i] -= freq[-1]

            indexes = individual != self.q
            Ts = self.T[indexes, :, individual[indexes]]
            useless_sensors = np.sum(Ts, axis=1)
            useless[i] = np.sum(useless_sensors == 0)
            f[i] = np.sum(Ts, axis=0)
        f = np.minimum(f, self.K)
        priority_factors = self.K
        return np.sum(priority_factors*np.square(f - self.K), axis=1) + useless_penalty*useless + activated_penalty*activated

    def solve(self, config: LSHADEConfig):
        # 1. Initialization
        np.random.seed(config.seed)
        random.seed(config.seed)
        population_size = config.population_size
        population = self.init_population(population_size=population_size, bounds=config.bounds)
        init_size = population_size
        m_cr = np.ones(config.memory_size)*0.5
        m_f = np.ones(config.memory_size)*0.5

        archive = []
        k = 0
        fitness = self.apply_fitness(population=population, useless_penalty=config.useless_penalty, activated_penalty=config.activated_penalty)

        all_indexes = np.arange(config.memory_size)
        current_generation = 0
        num_evals = population_size
        hist = []

        # Calculate max_iters
        n = population_size
        i = 0
        max_iters = 0
        while i < config.max_evals:
            max_iters += 1
            n = round((4 - init_size)/config.max_evals*i + init_size)
            i += n

        while num_evals < config.max_evals:
            # 2.1 Adaptation
            r = np.random.choice(all_indexes, population_size)
            cr = np.random.normal(m_cr[r], 0.1, population_size)
            cr = np.clip(cr, 0, 1)
            cr[m_cr[r] == 1] = 0
            f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=population_size)
            f[f > 1] = 0

            while sum(f <= 0) != 0:
                r = np.random.choice(all_indexes, sum(f <= 0))
                f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=sum(f <= 0))

            p = np.ones(population_size)*0.15

            # 2.2 Common steps
            mutated = apply_mutation(population, archive, fitness, f.reshape(len(f), 1), p, config.bounds)
            crossed = apply_crossover(population, mutated, cr.reshape(len(f), 1))
            c_fitness = self.apply_fitness(crossed, config.activated_penalty, config.activated_penalty)
            num_evals += population_size
            population, indexes = select(population, crossed, fitness, c_fitness, return_indexes=True)

            # 2.3 Adapt for next generation
            archive.extend(population[indexes])

            if len(indexes) > 0:
                if len(archive) > population_size:
                    archive = random.sample(archive, population_size)

                weights = np.abs(fitness[indexes] - c_fitness[indexes])
                weights /= np.sum(weights)
                m_cr[k] = np.sum(weights*cr[indexes]**2) / np.sum(weights*cr[indexes])
                if np.isnan(m_cr[k]):
                    m_cr[k] = 1
                m_f[k] = np.sum(weights*f[indexes]**2) / np.sum(weights*f[indexes])
                k += 1
                if k == config.memory_size:
                    k = 0

            fitness[indexes] = c_fitness[indexes]
            # Adapt population size
            new_population_size = round((4 - init_size) / config.max_evals*num_evals + init_size)
            if population_size > new_population_size:
                population_size = new_population_size
                best_indexes = np.argsort(fitness)[:population_size]
                population = population[best_indexes]
                fitness = fitness[best_indexes]
                if k == init_size:
                    k = 0
            best_fn = np.argmin(fitness)
            hist.append((population[best_fn], fitness[best_fn]))

        best = np.argmin(fitness)
        return population[best], fitness[best], hist
