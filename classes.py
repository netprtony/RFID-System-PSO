import numpy as np
from utils import fitness, calculate_covered_tags, fitness_function_basic, calculate_interference_basic, tentative_reader_elimination, calculate_coverage, calculate_inertia_weight  # Import from utils.py
GRID_X, GRID_Y = 25, 25  # Kích thước của lớp học
MOVE_PERCENTAGE_MIN = 0.01
MOVE_PERCENTAGE_MAX = 0.02

class Tags:
    def __init__(self, dim):
        self.position = np.random.rand(dim) * [GRID_X, GRID_Y]
        #self.id = np.random.randint(2001210000, 2001220000)
        #self.covered = False

    def update_position(self):
        move_distance = np.random.uniform(MOVE_PERCENTAGE_MIN * GRID_X, MOVE_PERCENTAGE_MAX * GRID_X)
        angle = np.random.rand() * 2 * np.pi  # Tạo góc ngẫu nhiên
        self.position += move_distance * np.array([np.cos(angle), np.sin(angle)])
        self.position = np.clip(self.position, [0, 0], [GRID_X, GRID_Y])

class Readers:
    def __init__(self, dim):
        self.position = np.random.rand(dim) * [GRID_X, GRID_Y]
        self.velocity = np.random.rand(dim) * [0, 0.1]
        self.best_position = self.position.copy()
        self.best_value = 0

    def update_velocity(self, global_best_position, w, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity

class SSPSO:
    def __init__(self, num_particles, dim, max_iter):
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.readers = [Readers(dim) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(-1, 1, dim)
        self.global_best_value = float('inf')

    def optimize(self, TAGS, RFID_RADIUS):
        for i in range(self.max_iter):
            j = 0
            print(f"Iteration {i + 1} ----------------------------------------------------------")
            for reader in self.readers:
                print(f"Reader {j + 1}")
                # Tính toán số thẻ được phủ sóng
                tags_covered = calculate_covered_tags(self.readers, TAGS, RFID_RADIUS)
            
                # Tính nhiễu
                interference = calculate_interference_basic(self.readers, TAGS, RFID_RADIUS)

                # Tính giá trị hàm mục tiêu
                #fitness_value = fitness_function_basic(tags_covered, interference)
                fitness_value = fitness(tags_covered, len(TAGS), interference, len(self.readers), len(self.readers))

                if fitness_value > reader.best_value:  # Tối ưu hóa
                    reader.best_position = reader.position.copy()
                    reader.best_value = fitness_value

                if fitness_value > self.global_best_value:  # Cập nhật vị trí tốt nhất toàn cục
                    self.global_best_position = reader.position.copy()
                    self.global_best_value = fitness_value
                w = calculate_inertia_weight(0.9 ,0.4, i, self.max_iter)
                reader.update_velocity(self.global_best_position, w)
                print(f"    Old position: {reader.position}")
                reader.update_position()
                print(f"    New position: {reader.position}")
                j+=1
                
            # # Sau khi hoàn thành một vòng lặp tối ưu hóa, áp dụng TRE
            # self.readers = tentative_reader_elimination(self.readers, tagS, 
            #                                         coverage_function=calculate_coverage(self.readers, tagS, RFID_RADIUS),
            #                                         max_recover_generations=5)   
