import numpy as np
import matplotlib.pyplot as plt

# Hàm mục tiêu: trong ví dụ này, ta dùng hàm đơn giản để minh họa (có thể thay đổi theo yêu cầu)
def objective_function(position):
    return np.sum(position**2)

# Lớp đại diện cho mỗi hạt trong PSO
class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity

class SPSO:
    def __init__(self, num_particles, dim, max_iter):
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.particles = [Particle(dim) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(-1, 1, dim)
        self.global_best_value = float('inf')
        self.history = []

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Đánh giá hàm mục tiêu
                fitness_value = objective_function(particle.position)
                # Cập nhật vị trí tốt nhất của từng hạt
                if fitness_value < particle.best_value:
                    particle.best_position = particle.position.copy()
                    particle.best_value = fitness_value
                # Cập nhật vị trí tốt nhất của toàn bộ quần thể
                if fitness_value < self.global_best_value:
                    self.global_best_position = particle.position.copy()
                    self.global_best_value = fitness_value
            # Cập nhật vận tốc và vị trí của mỗi hạt
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
            # Ghi lại giá trị tốt nhất tại mỗi vòng lặp
            self.history.append(self.global_best_value)

        return self.global_best_position, self.global_best_value