import numpy as np
import matplotlib.pyplot as plt
#import csv
import matplotlib.animation as animation
GRID_X, GRID_Y = 50, 50 # Kích thước của 1 lớp học
NUM_INDIVIDUALS = 500 # Sô lượng sinh viên
NUM_RFID_READERS = 100  # Số lượng đầu đọc RFID
NUM_ITERATION = 200  # Số vòng lặp
RFID_RADIUS = 3.69 # Bán kính vùng phủ sóng của đầu đọc
DIM = 2
ALPHA = 0.7
UPDATE_INTERVAL = 500
MOVE_PERCENTAGE_MIN = 0.01  # Tỷ lệ di chuyển tối thiểu
MOVE_PERCENTAGE_MAX = 0.02  # Tỷ lệ di chuyển tối đa

def schwefel(reader):
    reader = np.atleast_2d(reader)
    x, y = reader.shape
    res = 418.982887*y - np.sum(reader*np.sin(np.sqrt(np.absolute(reader))), axis=1, keepdims=True)
    assert(res.shape == (x, 1))
    return res
# Hàm để kiểm tra khoảng cách giữa hai điểm
def is_valid_distance(new_point, points, min_distance):
    for point in points:
        if np.linalg.norm(new_point - point) < min_distance:
            return False
    return True


def BieuDo(READERS, STUDENTS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')
    # Tối ưu hóa bằng thuật toán PSO
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION)
    r = sspso.optimize()
    print(r)
    for reader in READERS:
            reader.update_position()
    # Trích xuất vị trí sinh viên và đầu đọc RFID
    student_positions = np.array([student.position for student in STUDENTS])
    reader_positions = np.array([reader.position for reader in READERS])

    scatter_students = ax.scatter(student_positions[:, 0], student_positions[:, 1], color='blue', label='Students', s=10)
    scatter_rfid = ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='RFID Readers', marker='^')
    scatter_rfid.set_offsets(reader_positions)
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)
    # Text hiển thị số sinh viên trong vùng bán kính
    count_text = ax.text(0.02, 1.05, 'Students in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
   

    def update(frame):
        for student in STUDENTS:
            student.update_position()
        student_positions = np.array([student.position for student in STUDENTS])
        #reader_positions = np.array([reader.position for reader in READERS])

        # Cập nhật màu sắc của sinh viên dựa trên vị trí
        colors = []
        count_in_range  = 0
        for student in student_positions:
            # Kiểm tra khoảng cách đến các đầu đọc RFID
            if any(np.linalg.norm(student - reader.position) <= RFID_RADIUS for reader in READERS):
                colors.append('green')  # Đổi thành màu xanh lục
                count_in_range  += 1 
            else:
                colors.append('blue')  # Giữ màu xanh dương

        scatter_students.set_offsets(student_positions)
        scatter_students.set_color(colors)  # Cập nhật màu sắc

        scatter_students.set_offsets(student_positions)
        # scatter_rfid.set_offsets(reader_positions)

        # for i, circle in enumerate(circles):
        #     circle.center = reader_positions[i]
       # Cập nhật số sinh viên trong vùng bán kính
        count_text.set_text(f'Students in range: {count_in_range}')
        print(count_in_range)
        return scatter_students, *circles

    ani = animation.FuncAnimation(fig, update, frames=NUM_ITERATION, interval=UPDATE_INTERVAL, blit=True, repeat=False)
    plt.show()

class Students:
    def __init__(self, dim):
        self.position = np.random.rand(dim) * [GRID_X, GRID_Y]
        self.id = np.random.randint(2001210000, 2001220000)
       

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
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity

def fitness(position):
    epsilon = 1e-100
    fitness = 1/(schwefel(position) + epsilon)
    return fitness
class SSPSO:
    def __init__(self, num_particles, dim, max_iter, alpha=0.5):
        self.num_particles = num_particles
        self.dim = dim
        self.alpha = alpha
        self.max_iter = max_iter
        self.readers = [Readers(dim) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(-1, 1, dim)
        self.global_best_value = float('inf')

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.readers:
                # Đánh giá hàm mục tiêu
                fitness_value = fitness(particle.position)
                # Cập nhật vị trí tốt nhất của từng hạt
                if fitness_value < particle.best_value:
                    particle.best_position = particle.position.copy()
                    particle.best_value = fitness_value
                # Cập nhật vị trí tốt nhất của toàn bộ quần thể
                if fitness_value < self.global_best_value:
                    self.global_best_position = particle.position.copy()
                    self.global_best_value = fitness_value
            # Cập nhật vận tốc và vị trí của mỗi hạt
            for particle in self.readers:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
        final_positions = np.array([readers.position for readers in self.readers])
        return final_positions
        
            


readers =  [Readers(DIM) for _ in range(NUM_RFID_READERS)]
students = [Students(DIM) for _ in range(NUM_INDIVIDUALS)]
BieuDo(readers, students)
