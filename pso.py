import numpy as np
from data import Data
import typing
import random
from copy import deepcopy
from util import init_T, is_within_FoV

class Result(typing.NamedTuple):
    particle: np.ndarray
    fitness: float
    achieved_coverage: np.ndarray
    useless: int
    active: int

class PSOConfig(typing.NamedTuple):
    pop_size: int
    threshold: float
    useless_penalty: float
    active_penalty: float
    delta: int

class Particle():
    def __init__(self, states: np.ndarray, directions: np.ndarray):
        self.states = states
        self.directions = directions

    def copy(self):
        return Particle(self.states.copy(), self.directions.copy())


class PSO:
    def __init__(self, w=0.8, c1=0.25, c2=0.25) -> None:
        """
        Init PSO solver with PSO's hyperparameter
        """
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.name = 'PSO'
        self._compiled = False

    def adapt(self, data: Data):
        """
        Setup for solving a particular network, take in network data
        """
        self.n = data.n
        self.m = data.m
        self.q = data.q
        self.K = data.K
        self.T = init_T(data)

        # None value atribute
        self.POPULATION_SIZE = None
        self.threshold = None
        self._compiled = False

    def compile(self, config: PSOConfig):
        """
        Set evolutionary algorithm hyperparameter
        """
        self.POPULATION_SIZE = config.pop_size
        self.threshold = config.threshold
        self.useless_penalty = config.useless_penalty
        self.active_penalty = config.active_penalty
        self.delta = config.delta

        self._compiled = True

    def solve(self, init_type='uni', heu_init=0.5, max_gens=100, verbose=2):
        if not self._compiled:
            return RuntimeError("Model has not been compiled. Please execute PSO.compile method")
        if verbose > 2 or verbose < 0 or max_gens < 1:
            raise ValueError("Invalid verbose or negative max_gens at 'PSO.solve'")

        history = [] # save best particle's fitness through all gens

        # init particles
        X = self.init_particle(init_type=init_type, heu_init=heu_init)
        X_best = deepcopy(X)

        swarm_best_fitness = float('-inf')
        swarm_best_par = np.array([])
        for i in range(self.POPULATION_SIZE):
            particle_fitness = self.cal_fitness(X[i])
            if particle_fitness > swarm_best_fitness:
                swarm_best_fitness = particle_fitness
                swarm_best_par = X[i].copy()

        history.append(swarm_best_fitness)

        # init velocity
        V_state, V_dir = self.init_particle_velo(max_velo=1)

        # search loop
        gen = 1
        not_grow_gens = 0

        w = self.w
        c1 = self.c1
        c2 = self.c2

        while gen <= max_gens and not_grow_gens <= self.delta:
            grow = False

            swarm_best_par_w = self.particle_eval(swarm_best_par)

            new_swarm_best_par = None

            for i in range(self.POPULATION_SIZE):
                best_par_w = self.particle_eval(X_best[i])

                r1 = np.random.rand()
                r2 = np.random.rand()

                V_state[i] = (w*V_state[i] + c1*r1*best_par_w*(X_best[i].states - X[i].states)
                                    + c2*r2*swarm_best_par_w*(swarm_best_par.states - X[i].states))
                V_dir[i] = (w*V_dir[i] + c1*r1*best_par_w*(X_best[i].directions - X[i].directions)
                                    + c2*r2*swarm_best_par_w*(swarm_best_par.directions - X[i].directions))

                X[i].states = np.maximum(-1, np.minimum(1, X[i].states + V_state[i]))
                X[i].directions = (np.rint(X[i].directions + V_dir[i])).astype(int) % self.q

                particle_fitness = self.cal_fitness(X[i])
                if particle_fitness > self.cal_fitness(X_best[i]):
                    X_best[i] = X[i].copy()
                    if particle_fitness > swarm_best_fitness:
                        swarm_best_fitness = particle_fitness
                        new_swarm_best_par = X[i].copy()
                        grow = True

            history.append(swarm_best_fitness)
            gen += 1
            if grow:
                not_grow_gens = 0
                swarm_best_par = new_swarm_best_par
            else:
                not_grow_gens += 1

            # if not_grow_gens > self.delta or gen > max_gens:
            #     print(f'Gen {gen}: Stopped!')

        f, useless, active = self.achieved_coverage(swarm_best_par)
        result = Result(
            swarm_best_par,
            swarm_best_fitness,
            f,
            useless,
            active
        )

        return {'result': result,
                'history': history}


    def init_particle(self, init_type='uni', heu_init=0.5):
        particles = []
        if init_type == 'uni':
            for i in range(self.POPULATION_SIZE):
                particle = Particle(states=2*np.random.rand(self.n)-1, directions=np.random.randint(0, self.q, self.n))
                particles.append(particle)

        elif init_type == 'pre_deter_state':
            states = np.zeros(self.n)
            for i in range(self.n):
                if np.sum(self.T[i]) != 0:
                    states[i] = 1.0
                else:
                    states[i] = -1.0

            for i in range(self.POPULATION_SIZE):
                particle = Particle(states=states.copy(), directions=np.random.randint(0, self.q, self.n))
                particles.append(particle)

        elif init_type == 'heuristic':
            heuristic_size = int(self.POPULATION_SIZE*heu_init)
            rand_size = self.POPULATION_SIZE - heuristic_size

            heu_states = np.zeros(self.n)
            for i in range(self.n):
                if np.sum(self.T[i]) != 0:
                    heu_states[i] = 1.0
                else:
                    heu_states[i] = -1.0

            heu_directions = np.zeros((heuristic_size, self.n), dtype=int)
            count = np.zeros((self.n, self.q), dtype=int)
            for i in range(self.n):
                for p in range(self.q):
                    count[i, p] = np.sum(self.K*self.T[i, :, p])
            probs = np.exp(count)
            probs = probs/np.sum(probs, axis=1, keepdims=True)
            for i in range(self.n):
                heu_directions[:, i] = np.random.choice(np.arange(self.q), p=probs[i], size=heuristic_size)
            for i in range(heuristic_size):
                heu_particle = Particle(states=heu_states.copy(), directions=heu_directions[i])
                particles.append(heu_particle)

            for i in range(rand_size):
                particle = Particle(states=2*np.random.rand(self.n)-1, directions=np.random.randint(0, self.q, self.n))
                particles.append(particle)

        random.shuffle(particles)
        return particles

    def init_particle_velo(self, max_velo):
        state_velo = np.random.uniform(low=-0.2, high=0.2, size=(self.POPULATION_SIZE, self.n))
        direct_velo = np.random.uniform(low=-max_velo, high=max_velo, size=(self.POPULATION_SIZE, self.n))

        return state_velo, direct_velo

    def mutate(self, particle: Particle):
        mutated_particle = particle.copy()
        mutate_sensor = np.random.randint(low=0, high=self.n-1)
        mutated_particle.states[mutate_sensor] *= -1
        mutated_particle.directions[mutate_sensor] = np.random.randint(low=0, high=self.q)
        return mutated_particle

    def achieved_coverage(self, particle: Particle):
        f = np.zeros((self.m,), dtype=int)
        useless = 0
        active_sensor = (particle.states >= 0).sum()

        for i in range(self.n):
            if particle.states[i] >= 0:
                track = False
                for j in range(self.m):
                    if self.T[i, j, particle.directions[i]]:
                        track = True
                        f[j] += 1

                if not track:
                    useless += 1

        return f, useless, active_sensor

    def particle_eval(self, particle: Particle):
        sensor_eval = np.zeros(self.n)

        for i in range(self.n):
            target_in_range = np.sum(self.T[i])

            if target_in_range == 0:
                if particle.states[i] < 0:
                    sensor_eval[i] = 1
                else:
                    sensor_eval[i] = 0
            else:
                if particle.states[i] >= 0:
                    sensor_eval[i] = 0
                else:
                    sensor_eval[i] = np.sum(self.T[i, :, particle.directions[i]])
                sensor_eval[i] = sensor_eval[i]/target_in_range

        return np.exp(sensor_eval)

    def cal_fitness(self, particle: Particle):
        f, useless, active_sensors = self.achieved_coverage(particle)
        f = np.minimum(f, self.K)
        priority_factors = self.K
        return -(np.sum(priority_factors*np.square(f - self.K)) + self.useless_penalty*useless + self.active_penalty*active_sensors)


class DPSO(PSO):
    def __init__(self, chi=1.25, w=1.3, c1=0.45, c2=0.25, c3=0.15):
        super().__init__()
        self.chi = chi
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.name = 'DPSO'

    def solve(self, init_type='uni', heu_init=0.5, max_gens=100, verbose=2):
        if not self._compiled:
            return RuntimeError("Model has not been compiled. Please execute DPSO.compile method")
        if verbose > 2 or verbose < 0 or max_gens < 1:
            raise ValueError("Invalid verbose or negative max_gens at 'DPSO.solve'")

        history = []

        # init particles
        X = self.init_particle(init_type=init_type, heu_init=heu_init)
        X_best = deepcopy(X)

        swarm_best_fitness = float('-inf')
        swarm_best_par = np.array([])
        for i in range(self.POPULATION_SIZE):
            particle_fitness = self.cal_fitness(X[i])
            if particle_fitness > swarm_best_fitness:
                swarm_best_fitness = particle_fitness
                swarm_best_par = X[i].copy()

        history.append(swarm_best_fitness)

        # init velocity
        V_state, V_dir = self.init_particle_velo(max_velo=2)

        # search loop
        gen = 1
        not_grow_gens = 0

        chi = self.chi
        w = self.w
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3

        while gen <= max_gens and not_grow_gens <= self.delta:
            grow = False

            swarm_best_par_w = self.particle_eval(swarm_best_par)
            D_state, D_dir = self.cal_democratic(X)

            new_swarm_best_par = None

            for i in range(self.POPULATION_SIZE):
                best_par_w = self.particle_eval(X_best[i])
                X_i_eval = self.particle_eval(X[i])

                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()

                V_state[i] = chi*(w*np.exp(-X_i_eval)*V_state[i]+ c1*r1*best_par_w*(X_best[i].states - X[i].states)
                                    + c2*r2*swarm_best_par_w*(swarm_best_par.states - X[i].states)
                                    + c3*r3*D_state[i])
                V_dir[i] = chi*(w*np.exp(-X_i_eval)*V_dir[i] + c1*r1*best_par_w*(X_best[i].directions - X[i].directions)
                                    + c2*r2*swarm_best_par_w*(swarm_best_par.directions - X[i].directions)
                                    + c3*r3*D_dir[i])

                V_state[i] = V_state[i]*np.abs(V_dir[i])

                X[i].states = np.maximum(-1, np.minimum(1, X[i].states + V_state[i]))
                X[i].directions = (np.rint(X[i].directions + V_dir[i])).astype(int) % self.q

                particle_fitness = self.cal_fitness(X[i])
                particle_best_fitness = self.cal_fitness(X_best[i])
                if particle_fitness > particle_best_fitness:
                    X_best[i] = X[i].copy()

                if not_grow_gens > self.delta/2:
                    mutated_particle = self.mutate(X[i])
                    mutated_particle_fitness = self.cal_fitness(mutated_particle)
                    if mutated_particle_fitness > particle_best_fitness:
                        X[i] = mutated_particle
                        particle_best_fitness = mutated_particle_fitness

                if particle_best_fitness > swarm_best_fitness:
                    swarm_best_fitness = particle_best_fitness
                    new_swarm_best_par = X_best[i].copy()
                    grow = True

            history.append(swarm_best_fitness)
            gen += 1
            if grow:
                not_grow_gens = 0
                swarm_best_par = new_swarm_best_par
            else:
                not_grow_gens += 1

            if not_grow_gens > self.delta or gen > max_gens:
                print(f'Gen {gen}: Stopped!')

        f, useless, active = self.achieved_coverage(swarm_best_par)
        result = Result(
            swarm_best_par,
            swarm_best_fitness,
            f,
            useless,
            active
        )

        return {'result': result,
                'history': history}

    def cal_democratic(self, X):
        obj = np.array([-self.cal_fitness(X[i]) for i in range(self.POPULATION_SIZE)])
        Q = self.cal_democratic_weight(obj)

        D_state = np.zeros((self.POPULATION_SIZE, self.n))
        D_dir = np.zeros((self.POPULATION_SIZE, self.n))
        for i in range(self.POPULATION_SIZE):
            D_state[i] = sum(Q[i][k]*(X[k].states-X[i].states) for k in range(self.POPULATION_SIZE))
            D_dir[i] = sum(Q[i][k]*(X[k].directions-X[i].directions) for k in range(self.POPULATION_SIZE))

        return D_state, D_dir

    def cal_democratic_weight(self, obj):
        obj_best = np.min(obj)
        obj_worst = np.max(obj)

        E = np.zeros((self.POPULATION_SIZE, self.POPULATION_SIZE))
        elements = np.zeros_like(E)
        for i in range(self.POPULATION_SIZE):
            for k in range(self.POPULATION_SIZE):
                rand = np.random.rand()
                if (obj[k]-obj[i])/(obj_worst-obj_best) > rand or obj[k] < obj[i]:
                    E[i][k] = 1
                elements[i][k] = E[i][k]*(obj_best/obj[k])

        Q = elements.copy()
        for i in range(self.POPULATION_SIZE):
            Q[i] = Q[i]/np.sum(elements[i])

        return Q