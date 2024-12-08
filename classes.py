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
        self.max_velocity = max_velocity  # Thêm giới hạn tốc độ tối đa
        self.velocity = np.random.uniform(-self.max_velocity, self.max_velocity, size=dim)
        self.best_position = self.position.copy()
        self.best_value = 0
        self.active = True

    def update_velocity(self, global_best_position, w, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))

        cognitive_component = c1 * r1 * (self.best_position - self.position) 
        social_component = c2 * r2 * (global_best_position - self.position)

        # Tính toán tốc độ mới và ràng buộc nó theo giới hạn tối đa
        new_velocity = w * self.velocity + cognitive_component + social_component #Công thức cập nhật vận tốc (1)
        self.velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

    def update_position(self):
        self.position += self.velocity # Cập nhật vị trí mới (2)

        self.position[0] = np.clip(self.position[0], RFID_RADIUS, GRID_X - RFID_RADIUS)
        self.position[1] = np.clip(self.position[1], RFID_RADIUS, GRID_Y - RFID_RADIUS)

class ParticleSwarmOptimizationAlgorithm:
    def __init__(self, num_particles, dim, max_iter, readers):
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.readers = readers
        self.global_best_position = np.random.uniform(-1, 1, dim)
        self.global_best_value = 0
        self.best_positions = []
         # Biến đếm số vòng lặp fitness không đổi
        
    def optimize(self, TAGS, RFID_RADIUS=3.69):
        chaos_value = np.random.rand()  # Khởi tạo giá trị hỗn loạn ban đầu
        mu = 4  # Hằng số hỗn loạn
        stagnant_iterations = 0
        # print("Các đầu đọc ở vị trí ngẫu nhiên ban đầu")
        # for i, reader in enumerate(self.readers):
        #     if reader.active:
        #         print(f"Reader {i + 1} - Initial Position: {reader.position}")
        for i in range(self.max_iter):
            j = 0
            #print(f"Iteration {i + 1} ----------------------------{i + 1}------------------------{i + 1}")
             # Flag để kiểm tra nếu fitness_value không thay đổi trong vòng lặp này
            fitness_changed = False
            # Cập nhật giá trị hỗn loạn (14)
            chaos_value = mu * chaos_value * (1 - chaos_value)
            w = calculate_inertia_weight(0.9 ,0.4, i, self.max_iter) * (1 + chaos_value * 0.1)
            itr_stop = 0
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
                    w = w * (0.5 + chaos_value / 2)#(16)
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
            
            #fitness_tracking.append([i + 1, self.global_best_value])
            # Kiểm tra nếu fitness_value không thay đổi
            if fitness_changed:
                stagnant_iterations = 0  # Reset đếm nếu có thay đổi fitness
            else:
                stagnant_iterations += 1  # Tăng đếm nếu không có thay đổi

            # Dừng nếu fitness không thay đổi trong 5 vòng lặp liên tiếp hoặc đạt 100 vòng lặp
            if stagnant_iterations >= 5:
                itr_stop = i + 1
                print(Fore.RED +"Fitness không đổi trong 5 vòng lặp liên tiếp. Dừng tối ưu hóa.")
                break 
    
        print(Style.BRIGHT + f"Tại vòng lặp thứ {i + 1} đạt độ bao phủ :{COV}, độ nhiễu :{ITF}, giá trị fitness = {self.global_best_value} vị trí tốt nhất: {self.global_best_position}")
        # Trả về danh sách đầu đọc với vị trí tối ưu
        return self.readers, itr_stop, self.global_best_value
    
class Firefly:
    def __init__(self, position, dim=2, alpha=0.5, gamma=1, beta0=1):
        """
        Khởi tạo một đom đóm.
        """
        self.position = position                          # Vị trí hiện tại của đom đóm
        self.light_intensity = 0                          # Cường độ ánh sáng (giá trị hàm mục tiêu)
        self.alpha = alpha                                # Hệ số ngẫu nhiên
        self.gamma = gamma                                # Hệ số suy giảm ánh sáng
        self.beta0 = beta0                                # Sức hút ban đầu
        self.best_position = self.position.copy()         # Vị trí tốt nhất mà đom đóm đạt được
        self.active = True                                # Trạng thái hoạt động của đom đóm

    def move_towards(self, other, attractiveness, random_factor):
        """
        Đom đóm di chuyển về phía một đom đóm khác dựa trên sức hút.
        """
        direction = other.position - self.position
        self.position += attractiveness * direction + random_factor * (np.random.rand(len(self.position)) - 0.5)

class FireflyAlgorithm:
    def __init__(self, position, n_fireflies, dim = 2, bounds = [0 , GRID_X], alpha=0.5, gamma=1, beta0=1):
        self.fireflies = [Firefly(position[i]) for i in range(n_fireflies)]
        self.dim = dim
        self.bounds = bounds
        self.alpha = alpha
        self.gamma = gamma
        self.beta0 = beta0

    def _compute_attractiveness(self, distance):
        """
        Tính sức hút giữa hai đom đóm.
        """
        return self.beta0 * np.exp(-self.gamma * (distance ** 2))

    def optimize(self, TAGS, max_iter=100):
        """
        Chạy thuật toán FA để tối ưu hóa hàm mục tiêu với điều kiện dừng sớm.
        """
        # Biến lưu giá trị fitness tốt nhất và số vòng lặp không đổi
        best_fitness_value = -float('inf')
        stagnant_iterations = 0
        max_stagnant_iterations = 5  # Số vòng lặp liên tiếp cho phép không đổi

        for iter_count in range(max_iter):
            tracking_iter = iter_count + 1
            print(f"\nIteration {iter_count + 1}")
            for i in range(len(self.fireflies)):
                print(f"Firefly {i + 1}")
                for j in range(len(self.fireflies)):
                    if self.fireflies[j].light_intensity > self.fireflies[i].light_intensity:
                        distance = np.linalg.norm(self.fireflies[i].position - self.fireflies[j].position)
                        attractiveness = self._compute_attractiveness(distance)
                        random_factor = self.alpha * (np.random.rand(self.dim) - 0.5)
                        self.fireflies[i].move_towards(self.fireflies[j], attractiveness, random_factor)
                
                # Tính các giá trị liên quan
                COV = calculate_covered_tags(self.fireflies, TAGS, RFID_RADIUS)
                ITF = calculate_interference_basic(self.fireflies, TAGS, RFID_RADIUS)
                print(f"COV: {COV}, ITF: {ITF}")
                
                # Tính giá trị hàm mục tiêu
                fitness_value = fitness_function_basic(COV, ITF, TAGS, 0.8, 0.2)
                print(Fore.YELLOW + f"fitness value: {fitness_value}")
                
                # Cập nhật cường độ ánh sáng của đom đóm
                self.fireflies[i].light_intensity = fitness_value
            
            # Lấy giá trị fitness tốt nhất trong vòng lặp hiện tại
            current_best_fitness = max(f.light_intensity for f in self.fireflies)

            # Kiểm tra nếu fitness không đổi qua nhiều vòng lặp
            if np.isclose(current_best_fitness, best_fitness_value):
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0  # Reset nếu fitness thay đổi
                best_fitness_value = current_best_fitness

            # Điều kiện dừng sớm nếu fitness không đổi trong 5 vòng lặp liên tiếp
            if stagnant_iterations >= max_stagnant_iterations:
                print(Fore.RED + "Fitness không đổi trong 5 vòng lặp liên tiếp. Dừng tối ưu hóa.")
                break

        # Tìm đom đóm có cường độ ánh sáng tốt nhất
        best_firefly = max(self.fireflies, key=lambda f: f.light_intensity)
        return self.fireflies, tracking_iter, current_best_fitness