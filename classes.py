import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)
from utils import fitness_function_basic, calculate_covered_tags, constrain_velocity, calculate_interference_basic, calculate_inertia_weight, RFID_RADIUS# Import from utils.py
GRID_X, GRID_Y = 50, 50  # Kích thước của lớp học
MOVE_PERCENTAGE_MIN = 0.01
MOVE_PERCENTAGE_MAX = 0.02
class Tags:
    def __init__(self, position):
        self.position = position
        self.covered = False

    def update_position(self):
        move_distance = np.random.uniform(MOVE_PERCENTAGE_MIN * GRID_X, MOVE_PERCENTAGE_MAX * GRID_X)
        angle = np.random.rand() * 2 * np.pi  # Tạo góc ngẫu nhiên
        self.position += move_distance * np.array([np.cos(angle), np.sin(angle)])
        self.position = np.clip(self.position, [0, 0], [GRID_X, GRID_Y])

class Readers:
    def __init__(self, position, dim = 2, max_velocity=0.5):
        self.position = position
        self.velocity = np.random.rand(dim) * [0, 0.1]
        self.best_position = self.position.copy()
        self.best_value = 0
        self.max_velocity = max_velocity  # Thêm giới hạn tốc độ tối đa
        self.active = True

    def update_velocity(self, global_best_position, w, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))

        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)

        # Tính toán tốc độ mới và ràng buộc nó theo giới hạn tối đa
        new_velocity = w * self.velocity + cognitive_component + social_component
        self.velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        #self.velocity = constrain_velocity(self.velocity, GRID_Y, 0)
    def update_position(self):
        self.position += self.velocity

        self.position[0] = np.clip(self.position[0], 0, GRID_X)
        self.position[1] = np.clip(self.position[1], 0, GRID_Y)

class SSPSO:
    def __init__(self, num_particles, dim, max_iter, readers):
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.readers = readers
        self.global_best_position = np.random.uniform(-1, 1, dim)
        self.global_best_value = 0
        self.best_positions = []
         # Biến đếm số vòng lặp fitness không đổi
        
    def optimize(self, TAGS, RFID_RADIUS):
        chaos_value = np.random.rand()  # Khởi tạo giá trị hỗn loạn ban đầu
        mu = 4  # Hằng số hỗn loạn
        stagnant_iterations = 0
        fitness_tracking = []  # Mảng lưu số vòng lặp và giá trị fitness tại mỗi 5 vòng lặp
        print("Các đầu đọc ở vị trí ngẫu nhiên ban đầu")
        for i, reader in enumerate(self.readers):
            if reader.active:
                print(f"Reader {i + 1} - Initial Position: {reader.position}")
        for i in range(self.max_iter):
            j = 0
            print(f"Iteration {i + 1} ----------------------------{i + 1}------------------------{i + 1}")
             # Flag để kiểm tra nếu fitness_value không thay đổi trong vòng lặp này
            fitness_changed = False
            # Cập nhật giá trị hỗn loạn
            chaos_value = mu * chaos_value * (1 - chaos_value)
            w = calculate_inertia_weight(0.9 ,0.4, i, self.max_iter)
            
            for reader in self.readers:
                if reader.active:
                    print(f"Reader {j + 1}")
                    # Tính toán số thẻ được phủ sóng
                    COV = calculate_covered_tags(self.readers, TAGS, RFID_RADIUS)
                
                    # Tính nhiễu
                    ITF = calculate_interference_basic(self.readers, TAGS, RFID_RADIUS)
                    print(f"COV: {COV}, ITF: {ITF}")
                    # Tính giá trị hàm mục tiêu
                    fitness_value = fitness_function_basic(COV, ITF, TAGS, 0.8, 0.2)
                    print(Fore.YELLOW + f"fitness value: {fitness_value}")
                    w *=  chaos_value
                    if fitness_value > reader.best_value:  # Tối ưu hóa
                        reader.best_position = reader.position.copy()
                        reader.best_value = fitness_value
                        print(Fore.RED + f"Best position:{reader.best_position}, Best value:{reader.best_value}")

                    if fitness_value > self.global_best_value:  # Cập nhật vị trí tốt nhất toàn cục
                        self.global_best_position = reader.position.copy()
                        self.global_best_value = fitness_value
                        print(Fore.RED +f"Global best position{ self.global_best_position}, Global best value:{self.global_best_value}")
                        # Lưu vị trí của tất cả các đầu đọc khi đạt giá trị fitness tốt nhất
                        self.best_positions = [reader.position.copy() for reader in self.readers]
                        fitness_changed = True  # Đánh dấu có thay đổi fitness
                    reader.update_velocity(self.global_best_position, w, chaos_value)
                    print( Fore.GREEN +f"    Ví trí cũ : {reader.position}")
                
                    reader.update_position()
                    print(Fore.BLUE +f"    Ví trí mới : {reader.position}")
                    j+=1
            
              # Theo dõi giá trị fitness tại mỗi 5 vòng lặp
            #if (i + 1) % 5 == 0:
            fitness_tracking.append([i + 1, self.global_best_value])
            # Kiểm tra nếu fitness_value không thay đổi
            if fitness_changed:
                stagnant_iterations = 0  # Reset đếm nếu có thay đổi fitness
            else:
                stagnant_iterations += 1  # Tăng đếm nếu không có thay đổi

            # Dừng nếu fitness không thay đổi trong 5 vòng lặp liên tiếp hoặc đạt 100 vòng lặp
            if stagnant_iterations >= 5:
                print("Fitness không đổi trong 5 vòng lặp liên tiếp. Dừng tối ưu hóa.")
                break 
    
        print(Style.BRIGHT + f"Tại vòng lặp thứ {i + 1} đạt độ bao phủ :{COV}, độ nhiễu :{ITF}, giá trị fitness = {self.global_best_value} vị trí tốt nhất: {self.global_best_position}")
        # Trả về danh sách đầu đọc với vị trí tối ưu
        return self.readers, fitness_tracking
    
