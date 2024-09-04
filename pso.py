import numpy as np 
from deap import base, creator, tools, algorithms
from RFID  import calculate_distances, tags
import random

# Thiết lập môi trường PSO
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=None, smin=None, smax=None, best=None)

def generate_particle(size, smin, smax):
    particle = creator.Particle(random.uniform(-10, 10) for _ in range(size))
    particle.speed = [random.uniform(smin, smax) for _ in range(size)]
    particle.smin = smin
    particle.smax = smax
    return particle

def update_particle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(lambda u1, s: u1 * (s - x), u1, part.best)
    v_u2 = map(lambda u2, s: u2 * (s - x), u2, best)
    part.speed = list(map(lambda v, u1, u2: v + u1 + u2, part.speed, v_u1, v_u2))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(lambda x, v: x + v, part, part.speed))

def evaluate_particle(particle):
    new_readers = np.array(particle).reshape(-1, 2)
    distances = calculate_distances(tags, new_readers)
    return np.mean(distances),

toolbox = base.Toolbox()
toolbox.register("particle", generate_particle, size=8, smin=-1, smax=1)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", update_particle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", evaluate_particle)

# Tạo quần thể
population = toolbox.population(n=10)

# Tìm kiếm vị trí tốt nhất
best = None
for gen in range(100):
    for part in population:
        part.fitness.values = toolbox.evaluate(part)
        if not part.best or part.best.fitness > part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or best.fitness > part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values

    for part in population:
        toolbox.update(part, best)

# Kết quả tối ưu hóa
print("Vị trí tối ưu của các đầu đọc:")
optimal_readers = np.array(best).reshape(-1, 2)
print(optimal_readers)