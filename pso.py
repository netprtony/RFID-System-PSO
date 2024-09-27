import numpy as np
import matplotlib.pyplot as plt

def covered_ratio(rfid_readers, individuals):
    total_tags = len(individuals)
    covered_count = 0
    # Duyệt qua từng cá thể và kiểm tra nếu chúng nằm trong bán kính của bất kỳ đầu đọc nào
    for individual in individuals:
        if any(np.linalg.norm(individual - reader) <= RFID_RADIUS for reader in rfid_readers):
            covered_count += 1
    # Tính tỷ lệ phủ sóng
    coverage_ratio = covered_count / total_tags
    return coverage_ratio

def objective_function(position):
    return 1 / (1 + np.sum(position**2))

def BieuDo(READERS):
    # Khởi tạo biểu đồ
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)  
    ax.set_ylim(0, GRID_Y)  
    ax.set_aspect('equal', 'box')

    # Vẽ các cá thể
    ax.scatter(INDIVIDUALS[:, 0], INDIVIDUALS[:, 1], color='blue', label='Individuals')
    # Các danh sách để theo dõi các điểm cần vẽ

    scatter_rfid = ax.scatter(READERS[:, 0], READERS[:, 1], color='red', label='RFID Readers')
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in READERS]
    for circle in circles:
        ax.add_artist(circle)
    plt.show()
# Lớp đại diện cho mỗi hạt trong PSO
class Particle:
    def __init__(self, dim):
        self.position = np.random.rand(dim) * [GRID_X, GRID_Y]
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

class SSPSO:
    def __init__(self, num_particles, dim, max_iter, alpha=0.5):
        self.num_particles = num_particles
        self.dim = dim
        self.alpha = alpha
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
        # Trả về vị trí của từng hạt và vị trí tốt nhất toàn bộ quần thể
        final_positions = np.array([particle.position for particle in self.particles])
        return final_positions
    
GRID_X, GRID_Y = 100, 100 # Kích thước của 1 lớp học
NUM_INDIVIDUALS = 50 # Sô lượng sinh viên
NUM_RFID_READERS = 3  # Số lượng đầu đọc RFID
INDIVIDUALS = np.random.rand(NUM_INDIVIDUALS, 2) * [GRID_X, GRID_Y]
NUM_ITERATION = 200  # Số vòng lặp
RFID_RADIUS = 3.69 # Bán kính vùng phủ sóng của đầu đọc
DIM = 2
ALPHA = 0.7

sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, ALPHA)
final_positions = sspso.optimize()
print(final_positions)
BieuDo(final_positions)