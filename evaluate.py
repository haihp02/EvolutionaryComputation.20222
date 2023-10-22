import numpy as np
from tqdm import tqdm
from pso import *
from data import *
from matplotlib.pyplot import cm


def q_balancing_index(data: Data, sensors_mask):
    psi = achieved_coverage(data, sensors_mask)
    K = data.K
    QBI = (np.sum(psi) ** 3)/(np.sum(K)**3) * (np.sum(K**2)/np.sum(psi**2))
    return QBI

def achieved_coverage(data: Data, sensors_mask):
    m = data.m
    q = data.q
    T = init_T(data)
    K = data.K

    f = np.zeros((m, ), dtype=int)

    for i, p in enumerate(sensors_mask):
        if p is not None and p != q:
            for j in range(m):
                if T[i, j, p] and f[j] < K[j]:
                    f[j] += 1
    
    return f

def distance_index(data: Data, sensors_mask):
    K = data.K
    psi = achieved_coverage(data, sensors_mask)
    return 1 - np.sum((K-psi)*(K-psi))/np.sum(K*K)

def variance(data: Data, sensors_mask):
    K = data.K
    psi = achieved_coverage(data, sensors_mask)
    m = len(K)
    mk = np.array([np.sum(K == K[t]) for t in range(m)])
    
    uk = np.zeros_like(K, dtype=float)
    for t in range(m):
        uk[t] = np.sum(psi*(K == K[t]))/mk[t]

    return np.sum(np.square(psi-uk)/mk)

def activated_sensor(data: Data, particle: Particle=None, sensors_mask: np.ndarray=None, q=8):
    if particle is not None:
        return np.sum(particle.states >= 0)
    elif sensors_mask is not None:
        return np.sum(sensors_mask != q)
    else:
        return None

def coverage_quality(data: Data, sensors_mask: np.ndarray=None, particle: Particle=None):
    sensors = data.sensors
    targets = data.targets
    radius = data.radius
    n = data.n
    m = data.m
    q = data.q
    T = init_T(data)

    U = np.zeros((n, q, m), dtype=float)
    for i in range(n):
        for j in range(m):
            for p in range(q):
                if T[i, j, p]:
                    target = np.asarray(targets[j])
                    sensor = np.asarray(sensors[i])
                    v = target - sensor
                    U[i, p, j] = 1 - np.square(np.linalg.norm(v)/radius)

    S = np.zeros((n, q), dtype=bool)
    if particle is not None:
        for i in range(n):
            if particle.states[i] >= 0:
                S[i, particle.directions[i]] = True
    elif sensors_mask is not None:
        for i in range(n):
            if sensors_mask[i] != data.q:
                S[i, sensors_mask[i]] = True

    CQ = np.sum(np.sum(U, axis=2)*S)
    return CQ

def generate_metric_evaluation(datasets, res_set, data_type, data_size, metric, genome_type='sensors_mask', avg_for_multiple_datasets=True):
    metric_func_dict = {
        'distance_index': distance_index,
        'di': distance_index,
        'q_balancing_index': q_balancing_index,
        'qbi': q_balancing_index,
        # 'achieved_coverage': achieved_coverage,
        # 'ac': achieved_coverage,
        'coverage_quality': coverage_quality,
        'cq': coverage_quality,
        'variance': variance,
        'var':variance,
        'activated_sensor': activated_sensor,
        'as': activated_sensor,
    }

    metric_func = metric_func_dict[metric]
    
    metric_res = []
    for i, dataset in enumerate(res_set[data_type]):
        metric_res.append([metric_func(data=datasets[data_type][i][data_size][j], sensors_mask=res_set[data_type][i][data_size][j].sensors_mask)
                           for j in range(len(res_set[data_type][i][data_size]))])
    metric_res = np.array(metric_res)

    if avg_for_multiple_datasets:
        return metric_res.mean(axis=0)
    else:
        return metric_res
        

# def generate_metric_evaluation(dataset, model=DPSO(), data_type='fixed-sensor', size_type='small'):
#     DI = []
#     VAR = []
#     CQ = []
#     AS = []

#     if size_type == 'small':
#         pop_size = 200
#         delta = 100
#         max_gens = 500
#     elif size_type == 'large':
#         pop_size = 300
#         delta = 300
#         max_gens = 1000

#     for i in range(len(dataset)):
#         di = []
#         var = []
#         cq = []
#         activated_sensor = []
#         for data in tqdm(dataset[data_type][i][size_type]):
#             model.adapt(data)
#             config = Config(pop_size=pop_size, temperature=1000, threshold=0.7, useless_penalty=8.0, active_penalty=1.0, delta=delta)
#             model.compile(config)
#             result = model.solve(init_type='heuristic', heu_init=0.4, max_gens=max_gens, verbose=0)

#             di_score = distance_index(data.K, result['result'].achieved_coverage)
#             var_score = variance(data.K, result['result'].achieved_coverage)
#             cq_score = coverage_quality(result['result'].particle, data)
#             activated_sensor_score = activated_sensor(result['result'].particle)

#             di.append(di_score)
#             var.append(var_score)
#             cq.append(cq_score)
#             activated_sensor.append(activated_sensor_score)

#         DI.append(di)
#         VAR.append(var)
#         CQ.append(cq)
#         AS.append(activated_sensor)

#     return DI, VAR, CQ, AS

def plot_metric_evaluation(x_axis, y_axis, x_label, y_label, labels, title, figsize=(10, 8)):
    y_axis = list(y_axis)
    labels = list(labels)
    colors = cm.get_cmap('Accent').colors

    f, ax = plt.subplots(figsize=figsize)
    for y, label,color in zip(y_axis, labels, colors):
        ax.plot(x_axis, y, '-', linewidth=1.0, c=color, label=label)
        ax.plot(x_axis, y, 'o', markersize=3.0, c=color)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    return ax

def metrics_evaluation_visualization(DI, VAR, CQ, AS, model_name):

    DI_avg = np.mean(DI, axis=0)
    VAR_avg = np.mean(VAR, axis=0)
    AS_avg = np.mean(AS, axis=0)
    CQ_avg = np.mean(CQ, axis=0)

    x_axis = np.arange(len(DI_avg))
    figure, axis = plt.subplots(nrows=4, ncols=1, sharex=False, figsize=(15, 21))

    axis[0].set_title('Distance index')
    axis[0].plot(x_axis, DI_avg, 'r-', linewidth=1.0)
    axis[0].plot(x_axis, DI_avg, 'rs', markersize=3.0, label=model_name)
    axis[0].legend()

    axis[1].set_title('Variance')
    axis[1].plot(x_axis, VAR_avg, 'r-', linewidth=1.0)
    axis[1].plot(x_axis, VAR_avg, 'rs', markersize=3.0, label=model_name)
    axis[1].legend()

    axis[2].set_title('Activated sensor')
    axis[2].plot(x_axis, AS_avg, 'r-', linewidth=1.0)
    axis[2].plot(x_axis, AS_avg, 'rs', markersize=3.0, label=model_name)
    axis[2].legend()


    axis[3].set_title('Coverage quality')
    axis[3].plot(x_axis, CQ_avg, 'r-', linewidth=1.0)
    axis[3].plot(x_axis, CQ_avg, 'rs', markersize=3.0, label=model_name)
    axis[3].legend()


    plt.xlabel('Number of targets')
    plt.show()