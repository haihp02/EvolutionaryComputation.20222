import typing
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as ptc

class Data(typing.NamedTuple):
    n: int
    m: int
    q: int
    K: np.array
    radius: float
    sensors: np.array
    targets: np.array
    area: tuple

class SolveResult:
    sensors_mask = None
    particle = None

    def __init__(self, sensors_mask=None, particle=None):
        self.sensors_mask = sensors_mask
        self.particle = particle
    def __str__(self):
        if self.sensors_mask is not None:
            return self.sensors_mask
        elif self.particle is not None:
            return self.particle.states, self.particle.directions

def center_square(area):
    x_ = area[0]
    y_ = area[1]
    x_range = x_[1] - x_[0]
    y_range = y_[1] - y_[0]

    x = (x_[0] + x_range*0.25, x_[0] + x_range*0.75)
    y = (y_[0] + y_range*0.25, y_[0] + y_range*0.75)
    return (x, y)

def zipf(n, area):
    n_in = int(0.75*n)
    n_out = n - n_in

    points_out = np.random.rand(2, n_out)

    # map x and y values between minx - maxx, miny - maxy
    points_out[0, :] = np.interp(points_out[0, :], [0, 1], area[0])
    points_out[1, :] = np.interp(points_out[1, :], [0, 1], area[1])

    points_in = np.random.rand(2, n_in)
    area_in = center_square(area)

    # map x and y values between minx - maxx, miny - maxy
    points_in[0, :] = np.interp(points_in[0, :], [0, 1], area_in[0])
    points_in[1, :] = np.interp(points_in[1, :], [0, 1], area_in[1])

    points = np.concatenate((points_in, points_out), axis=1).T
    indicies = np.arange(n)
    np.random.shuffle(indicies)
    return points[indicies]

def random_points(n, area, distribution_type='uniform'):
    points = None

    if distribution_type == 'uniform':
        # random values between 0 - 1
        points = np.random.rand(2, n)

        # map x and y values between minx - maxx, miny - maxy
        points[0, :] = np.interp(points[0, :], [0, 1], area[0])
        points[1, :] = np.interp(points[1, :], [0, 1], area[1])

        points = points.T
    elif distribution_type == 'zipf':
        points = zipf(n, area)

    return points
    
def create_example_data(n_sensors, n_targets, max_k=3, sensing_radius=20, area=((0, 150), (0, 150)), distribution_type='uniform'):
    sensors = random_points(n_sensors, area, distribution_type)
    targets = random_points(n_targets, area, distribution_type)
    K = np.random.randint(low=1, high=max_k+1, size=n_targets)

    return Data(n=n_sensors,
                m=n_targets,
                q=8,
                K=K,
                radius=sensing_radius,
                sensors=sensors,
                targets=targets,
                area=area)

def create_example_smooth_data(n_sensors, n_targets, sensing_radius=20, area=((0, 150), (0, 150)), n_sub_areas=4, distribution_type='uniform'):
    split = int(np.sqrt(n_sub_areas))
    n_sub_areas = split**2

    n_sensors_parea = int(n_sensors/n_sub_areas)
    n_sensors_remain = n_sensors - n_sub_areas*n_sensors_parea
    n_targets_parea = int(n_targets/n_sub_areas)
    n_targets_remain = n_targets - n_sub_areas*n_targets_parea

    sub_area_w = (area[0][1] - area[0][0])/split
    sub_area_h = (area[1][1] - area[1][0])/split

    sensors = random_points(n_sensors_remain, area, distribution_type)
    targets = random_points(n_targets_remain, area, distribution_type)
    K = np.random.randint(low=1, high=4, size=n_targets_remain)
    
    for i in range(split):
        for j in range(split):
            sub_area = ((sub_area_w*i, sub_area_w*(i+1)), (sub_area_h*j, sub_area_h*(j+1)))
            sub_area_sensors = random_points(n_sensors_parea, sub_area, distribution_type)
            sub_area_targets = random_points(n_targets_parea, sub_area, distribution_type)
            sub_area_K = np.random.randint(low=1, high=4, size=n_targets_parea)

            sensors = np.append(sensors, sub_area_sensors, axis=0)
            targets = np.append(targets, sub_area_targets, axis=0)
            K = np.append(K, sub_area_K, axis=0)

    zipped = list(zip(targets, K))
    random.shuffle(zipped)
    targets, K = zip(*zipped)
    targets = np.array(targets)
    K = np.array(K)
    np.random.shuffle(sensors)
            
    return Data(n=n_sensors,
                m=n_targets,
                q=8,
                K=K,
                radius=sensing_radius,
                sensors=sensors,
                targets=targets,
                area=area)

def create_dataset(distribution_type='uniform'):
    sensing_radius = 20
    # Fixed_sensors
    fs_data = {'small': [], 'large': [], 'huge': []}
    # small
    data = create_example_smooth_data(n_sensors=30, n_targets=120, sensing_radius=sensing_radius, area=((0, 200), (0, 200)), n_sub_areas=10, distribution_type=distribution_type)
    n = data.n
    area = data.area
    sensors = data.sensors
    targets_all = data.targets
    K_all = data.K
    for i in range(5, 121, 5):
        targets = targets_all[:i]
        K = K_all[:i]
        fs_data['small'].append(Data(n=n,
                                     m=i,
                                     q=8,
                                     K=K,
                                     radius=sensing_radius,
                                     sensors=sensors,
                                     targets=targets,
                                     area=area))

    # large
    data = create_example_smooth_data(n_sensors=45, n_targets=180, sensing_radius=sensing_radius, area=((0, 300), (0, 300)), n_sub_areas=16, distribution_type=distribution_type)
    n = data.n
    area = data.area
    sensors = data.sensors
    targets_all = data.targets
    K_all = data.K
    for i in range(5, 181, 5):
        targets = targets_all[:i]
        K = K_all[:i]
        fs_data['large'].append(Data(n=n,
                                     m=i,
                                     q=8,
                                     K=K,
                                     radius=sensing_radius,
                                     sensors=sensors,
                                     targets=targets,
                                     area=area))
        
    # huge
    data = create_example_smooth_data(n_sensors=75, n_targets=300, sensing_radius=sensing_radius, area=((0, 500), (0, 500)), n_sub_areas=25, distribution_type=distribution_type)
    n = data.n
    area = data.area
    sensors = data.sensors
    targets_all = data.targets
    K_all = data.K
    for i in range(10, 301, 10):
        targets = targets_all[:i]
        K = K_all[:i]
        fs_data['huge'].append(Data(n=n,
                                    m=i,
                                    q=8,
                                    K=K,
                                    radius=sensing_radius,
                                    sensors=sensors,
                                    targets=targets,
                                    area=area))


    # Fixed_targets
    ft_data = {'small': [], 'large': [], 'huge': []}
    # small
    data = create_example_smooth_data(n_sensors=120, n_targets=30, sensing_radius=sensing_radius, area=((0, 200), (0, 200)), n_sub_areas=10, distribution_type=distribution_type)
    m = data.m
    area = data.area
    sensors_all = data.sensors
    targets = data.targets
    K = data.K
    for i in range(5, 121, 5):
        sensors = sensors_all[:i]
        ft_data['small'].append(Data(n=i,
                                     m=m,
                                     q=8,
                                     K=K,
                                     radius=sensing_radius,
                                     sensors=sensors,
                                     targets=targets,
                                     area=area))

    # large
    data = create_example_smooth_data(n_sensors=180, n_targets=45, sensing_radius=sensing_radius, area=((0, 300), (0, 300)), n_sub_areas=16, distribution_type=distribution_type)
    m = data.m
    area = data.area
    sensors_all = data.sensors
    targets = data.targets
    K = data.K
    for i in range(5, 181, 5):
        sensors = sensors_all[:i]
        ft_data['large'].append(Data(n=i,
                                     m=m,
                                     q=8,
                                     K=K,
                                     radius=sensing_radius,
                                     sensors=sensors,
                                     targets=targets,
                                     area=area))
    
    # huge
    data = create_example_smooth_data(n_sensors=300, n_targets=75, sensing_radius=sensing_radius, area=((0, 500), (0, 500)), n_sub_areas=25, distribution_type=distribution_type)
    m = data.m
    area = data.area
    sensors_all = data.sensors
    targets = data.targets
    K = data.K
    for i in range(10, 301, 10):
        sensors = sensors_all[:i]
        ft_data['huge'].append(Data(n=1,
                                    m=m,
                                    q=8,
                                    K=K,
                                    radius=sensing_radius,
                                    sensors=sensors,
                                    targets=targets,
                                    area=area))

    return {
        'fixed-sensor': fs_data,
        'fixed-target': ft_data,
    }

def create_multi_datasets(n_dataset=10, distributrion_type='uniform'):
    datasets = {'fixed-sensor': [], 'fixed-target': []}
    for i in range(n_dataset):
        dataset = create_dataset(distribution_type=distributrion_type)
        datasets['fixed-sensor'].append(dataset['fixed-sensor'])
        datasets['fixed-target'].append(dataset['fixed-target'])
    
    return datasets

def make_pan_boundaries(centroid, q, radius):
    # first line always lies on x_axis
    ans = []
    theta = 2*np.pi/q
    for i in range(q):
        x = radius*np.cos(theta*i)
        y = radius*np.sin(theta*i)
        ans.append((centroid[0] + x, centroid[1] + y))

    return ans

def show_network(data: Data, sensors_mask=None, particle=None, figsize: tuple=None):
    n = data.n
    m = data.m
    q = data.q
    K = data.K
    radius = data.radius
    sensors = data.sensors
    targets = data.targets
    margin = ((data.area[0][0]-10, data.area[0][1]+10), (data.area[1][0]-10, data.area[1][1]+10))
    theta = 360.0/q

    if figsize is not None:
        plt.figure(figsize=figsize)

    # plot sensors
    plt.plot(sensors[:, 0], sensors[:, 1], 'go', label='sensors')

    # plot targets with different size by target's importance
    plt.scatter(x=targets[K==1][:, 0], y=targets[K==1][:, 1], marker='^', c='red', s=40, label='targets')
    plt.scatter(x=targets[K==2][:, 0], y=targets[K==2][:, 1], marker='^', c='red', s=80, label='targets')
    plt.scatter(x=targets[K==3][:, 0], y=targets[K==3][:, 1], marker='^', c='red', s=120, label='targets')
    plt.scatter(x=targets[K==4][:, 0], y=targets[K==4][:, 1], marker='^', c='red', s=160, label='targets')

    ax = plt.gca()

    # draw sensor radius if active and direction if set
    for i in range(n):
        sensor = sensors[i]
        active = True

        if particle is not None:
            active = particle.states[i] >= 0
            if active:
                dir = particle.directions[i]
                theta1, theta2 = theta*dir, theta*(dir+1)
                wedge = ptc.Wedge(sensors[i], radius, theta1, theta2, color='#98f797', alpha=0.45)
                ax.add_artist(wedge)
        if sensors_mask is not None:
            active = sensors_mask[i] != q
            if active:
                dir = sensors_mask[i]
                theta1, theta2 = theta*dir, theta*(dir+1)
                wedge = ptc.Wedge(sensors[i], radius, theta1, theta2, color='#98f797', alpha=0.45)
                ax.add_artist(wedge)
        if active:
            circle = plt.Circle(sensor, radius, color='m', fill=False, linewidth=0.5)
            ax.add_artist(circle)
            pan_boundaries = make_pan_boundaries(sensor, q, radius)
            for point in pan_boundaries:
                plt.plot([sensor[0], point[0]], [sensor[1], point[1]], 'b--', alpha=0.2)

    plt.xlim(margin[0])
    plt.ylim(margin[1])
    ax.set_aspect(1.0)  # make aspect ratio square

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()

if __name__ == '__main__':
    N_SENSORS = 30
    N_TARGETS = 45
    network = create_example_data(N_SENSORS, N_TARGETS)
    show_network(network, figsize=(10, 10))