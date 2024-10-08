import numpy as np
import matplotlib.pyplot as plt
#import csv
import matplotlib.animation as animation
GRID_X, GRID_Y = 25, 25 # Kích thước của 1 lớp học
NUM_INDIVIDUALS = 700 # Sô lượng sinh viên
NUM_RFID_READERS = 35  # Số lượng đầu đọc RFID
NUM_ITERATION = 50  # Số vòng lặp
RFID_RADIUS = 3.69 # Bán kính vùng phủ sóng của đầu đọc
DIM = 2
ALPHA = 0.7
UPDATE_INTERVAL = 500
MOVE_PERCENTAGE_MIN = 0.01  # Tỷ lệ di chuyển tối thiểu
MOVE_PERCENTAGE_MAX = 0.02  # Tỷ lệ di chuyển tối đa

def remove_unnecessary_readers(students, readers, rfid_radius=3.69, coverage_threshold=0.98):
    """
    Loại bỏ các đầu đọc không cần thiết mà vẫn đảm bảo độ bao phủ tối ưu.
    
    Parameters:
    - students: Danh sách các sinh viên (vị trí của sinh viên)
    - readers: Danh sách các đầu đọc (vị trí của đầu đọc)
    - rfid_radius: Bán kính bao phủ của mỗi đầu đọc RFID
    - coverage_threshold: Ngưỡng tối thiểu của độ bao phủ (tỷ lệ sinh viên được bao phủ chấp nhận được)

    Returns:
    - Danh sách các đầu đọc tối ưu sau khi loại bỏ các đầu đọc không cần thiết
    """
    # Tính tổng số sinh viên
    total_students = len(students)
    
    # Bắt đầu từ tất cả các đầu đọc
    optimal_readers = readers.copy()
    
    # Tính độ bao phủ ban đầu
    initial_coverage = calculate_covered_students(students, optimal_readers, rfid_radius)
    print(f"Initial coverage: {initial_coverage * 100:.2f}%")
    
    # Kiểm tra từng đầu đọc để xem có thể loại bỏ được không
    for reader in readers:
        # Tạm thời loại bỏ đầu đọc này
        temp_readers = [r for r in optimal_readers if r != reader]
        
        # Tính độ bao phủ sau khi loại bỏ đầu đọc
        new_coverage = calculate_covered_students(students, temp_readers, RFID_RADIUS)
        
        # Nếu độ bao phủ sau khi loại bỏ đầu đọc vẫn trên ngưỡng cho phép, thì loại bỏ đầu đọc
        if new_coverage >= coverage_threshold * initial_coverage:
            print(f"Removing reader at {reader.position} - New coverage: {new_coverage * 100:.2f}%")
            optimal_readers = temp_readers
        else:
            print(f"Keeping reader at {reader.position} - Coverage would drop to {new_coverage * 100:.2f}%")
    
    return optimal_readers
# Hàm để kiểm tra khoảng cách giữa hai điểm
def optimize_rfid_placement_with_removal(students, initial_readers, max_iterations=100, rfid_radius=3.69, coverage_threshold=0.98):
    """
    Thuật toán tối ưu hóa với loại bỏ các đầu đọc không cần thiết để tối ưu độ bao phủ.
    
    Parameters:
    - students: Danh sách các sinh viên
    - initial_readers: Danh sách các đầu đọc ban đầu
    - max_iterations: Số lần lặp tối đa của thuật toán tối ưu hóa
    - rfid_radius: Bán kính bao phủ của mỗi đầu đọc
    - coverage_threshold: Ngưỡng tối thiểu của độ bao phủ
    
    Returns:
    - optimal_readers: Danh sách đầu đọc tối ưu
    """
    # Bắt đầu với tập đầu đọc ban đầu
    readers = initial_readers.copy()
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Chạy một bước của thuật toán tối ưu (ví dụ: tối ưu hóa bầy đàn PSO hoặc thuật toán khác)
        # Tại đây bạn có thể thêm quá trình cải thiện hoặc tối ưu thêm vị trí của các đầu đọc
        
        # Sau mỗi vài vòng lặp (ví dụ 10), loại bỏ các đầu đọc không cần thiết
        if iteration % 10 == 0:
            readers = remove_unnecessary_readers(students, readers, rfid_radius, coverage_threshold)
        
        # In kết quả độ bao phủ hiện tại
        coverage = calculate_covered_students(students, readers, rfid_radius)
        print(f"Coverage after iteration {iteration + 1}: {coverage * 100:.2f}%")
        
        # Nếu độ bao phủ đạt yêu cầu, có thể dừng sớm
        if coverage >= coverage_threshold:
            break
    
    # Trả về danh sách các đầu đọc tối ưu
    return readers
def is_valid_distance(new_point, points, min_distance):
    for point in points:
        if np.linalg.norm(new_point - point) < min_distance:
            return False
    return True
def calculate_inertia_weight(w_max, w_min, iter, iter_max):
    """
    Tính giá trị w theo công thức w = w_max - ((w_max - w_min) / iter_max) * iter
    
    Parameters:
    - w_max: Trọng số lớn nhất (w_max)
    - w_min: Trọng số nhỏ nhất (w_min)
    - iter: Số vòng lặp hiện tại
    - iter_max: Số vòng lặp tối đa

    Returns:
    - w: Trọng số w ở vòng lặp hiện tại
    """
    w = w_max - ((w_max - w_min) / iter_max) * iter
    return w
def calculate_covered_students(students, readers, rfid_radius = 3.69):
    """
    Tính số sinh viên được bao phủ bởi ít nhất một đầu đọc RFID.
    
    Parameters:
    - students: Danh sách các đối tượng sinh viên (vị trí của sinh viên)
    - readers: Danh sách các đối tượng đầu đọc (vị trí của đầu đọc)
    - rfid_radius: Bán kính bao phủ của mỗi đầu đọc RFID
    
    Returns:
    - Số lượng sinh viên được bao phủ
    """
    covered_students = 0
    
    for student in students:
        student_covered = False  # Giả định ban đầu sinh viên không được bao phủ
        
        for reader in readers:
            distance = np.linalg.norm(student.position - reader.position)
            
            # Kiểm tra xem sinh viên có nằm trong vùng bán kính của đầu đọc nào không
            if distance <= rfid_radius:
                student_covered = True
                break  # Dừng kiểm tra nếu sinh viên đã được bao phủ bởi ít nhất một đầu đọc
        
        if student_covered:
            covered_students += 1
    
    return covered_students/len(students)
def calculate_overlap(students, readers, rfid_radius):
    """
    Tính độ trùng lắp giữa các đầu đọc RFID.
    
    Parameters:
    - students: Danh sách các điểm cần bao phủ (tọa độ sinh viên)
    - readers: Danh sách tọa độ của các đầu đọc RFID
    - rfid_radius: Bán kính phủ sóng của mỗi đầu đọc
    
    Returns:
    - overlap_percentage: Tỉ lệ trùng lắp giữa các đầu đọc
    """
    total_students_covered = set()  # Set để lưu tất cả sinh viên được bao phủ ít nhất một lần
    overlapping_students = set()  # Set để lưu các sinh viên bị bao phủ bởi nhiều hơn 1 đầu đọc

    # Duyệt qua từng sinh viên để kiểm tra xem có được bao phủ bởi nhiều đầu đọc không
    for student in students:
        readers_covering_student = 0
        
        # Duyệt qua từng đầu đọc để xem nó có bao phủ sinh viên này không
        for reader in readers:
            if calculate_distance(student, reader) <= rfid_radius:
                readers_covering_student += 1
        
        # Nếu có ít nhất 1 đầu đọc bao phủ, thêm vào danh sách sinh viên được bao phủ
        if readers_covering_student > 0:
            total_students_covered.add(student)
        
        # Nếu có nhiều hơn 1 đầu đọc bao phủ, thêm vào danh sách sinh viên bị trùng lắp
        if readers_covering_student > 1:
            overlapping_students.add(student)

    # Tính tỉ lệ trùng lắp (số lượng sinh viên trùng lắp trên tổng số sinh viên được bao phủ)
    if len(total_students_covered) == 0:
        overlap_percentage = 0.0  # Tránh chia cho 0
    else:
        overlap_percentage = len(overlapping_students) / len(total_students_covered)
    
    return overlap_percentage
def BieuDoReader(READERS, STUDENTS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Trích xuất vị trí sinh viên và đầu đọc RFID
    student_positions = np.array([student.position for student in STUDENTS])
    reader_positions = np.array([reader.position for reader in READERS])
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION)
    ax.scatter(student_positions[:, 0], student_positions[:, 1], color='blue', label='Students', s=10)
    scatter_rfid = ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='RFID Readers', marker='^')

    # Tạo các hình tròn biểu diễn vùng phủ sóng của các đầu đọc RFID
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)

    # Text hiển thị số sinh viên trong vùng bán kính
    count_text = ax.text(0.02, 1.05, 'Students in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    max_no_change_iterations = 5  # Số lần không thay đổi tối đa trước khi dừng
    no_change_counter = 0         # Bộ đếm cho số lần không thay đổi liên tiếp
    previous_gbest_value = sspso.global_best_value  # Biến lưu giá trị của gbest từ lần chạy trước
    
    # Hàm cập nhật vị trí của các reader
    def update_reader(frame):
        global previous_gbest_value
        count_in_range = 0
        for student in student_positions:
            if any(np.linalg.norm(student - reader.position) <= RFID_RADIUS for reader in READERS):
                count_in_range += 1  

        count_text.set_text(f'Students in range: {count_in_range}')
        print(f"Iteration {frame}: {count_in_range} students in range")

        sspso.optimize(STUDENTS)
        if  sspso.global_best_value == previous_gbest_value:
            no_change_counter += 1  # Tăng bộ đếm nếu gbest_score không thay đổi
        else:
            no_change_counter = 0   # Reset bộ đếm nếu có thay đổi
            previous_gbest_value = sspso.global_best_value  # Cập nhật gbest_score mới nhất
       
        # Dừng vòng lặp nếu qua 5 lần không có sự thay đổi
        if no_change_counter >= max_no_change_iterations:
            print(f"Stopping early after iterations due to no change in objective function.")    

        for reader in READERS:
            w = calculate_inertia_weight(0.9 ,0.4, frame, NUM_ITERATION)
            reader.update_velocity(sspso.global_best_position, w)
            reader.update_position()
        
        # Cập nhật vị trí của các đầu đọc RFID
        reader_positions = np.array([reader.position for reader in READERS])
        
        scatter_rfid.set_offsets(reader_positions)
        
         # Cập nhật vị trí của các hình tròn vùng phủ sóng
        for i, circle in enumerate(circles):
            circle.center = reader_positions[i]

        return scatter_rfid, *circles
    # Animation cho quá trình cập nhật vị trí của reader
    ani_reader = animation.FuncAnimation(fig, update_reader, frames=NUM_ITERATION, interval=UPDATE_INTERVAL, blit=False, repeat=False)
    plt.show()
def BieuDoStudents(READERS, STUDENTS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')
    student_positions = np.array([student.position for student in STUDENTS])
    reader_positions = np.array([reader.position for reader in READERS])
    scatter_student = ax.scatter(student_positions[:, 0], student_positions[:, 1], color='blue', label='Students', s=10)
    ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='RFID Readers', marker='^')
    # Tạo các hình tròn biểu diễn vùng phủ sóng của các đầu đọc RFID
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)
     # Text hiển thị số sinh viên trong vùng bán kính
    count_text = ax.text(0.02, 1.05, 'Students in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    # Hàm cập nhật vị trí của các student
    def update_student(frame):
        for student in STUDENTS:
            student.update_position()
        student_positions = np.array([student.position for student in STUDENTS])
        student_positions = np.atleast_2d(student_positions)

        # Cập nhật màu sắc của sinh viên dựa trên vị trí
        colors = []
        count_in_range = 0
        for student in student_positions:
            if any(np.linalg.norm(student - reader.position) <= RFID_RADIUS for reader in READERS):
                colors.append('green')  # Đổi thành màu xanh lục nếu trong vùng phủ sóng
                count_in_range += 1
            else:
                colors.append('blue')  # Giữ màu xanh dương nếu ngoài vùng phủ sóng

        scatter_student.set_offsets(student_positions)
        scatter_student.set_color(colors)  # Cập nhật màu sắc của sinh viên

        # Cập nhật số sinh viên trong vùng bán kính
        count_text.set_text(f'Students in range: {count_in_range}')
        print(f"Iteration {frame}: {count_in_range} students in range")

        return scatter_student
    # Animation cho quá trình cập nhật vị trí của student
    ain = animation.FuncAnimation(fig, update_student, frames=999999, interval=UPDATE_INTERVAL, blit=False, repeat=False)
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
        self.best_value = 0

    def update_velocity(self, global_best_position, w, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        
        # Lấy vị trí tốt nhất toàn cục và vị trí hiện tại của từng hạt
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        # Cập nhật vận tốc
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity


def fitness(students_covered, total_students, overlap_points, total_readers, readers_used):
    """
    Tính toán fitness của một particle dựa trên số học sinh bao phủ, số reader sử dụng, và diện tích trùng lặp.
    
    Parameters:
    - students_covered: Số học sinh được bao phủ bởi các đầu đọc
    - total_students: Tổng số học sinh trong vùng
    - total_readers: Tổng số đầu đọc có thể đặt trong vùng
    - overlap_points: Số điểm trùng lặp (số học sinh bị nhiều đầu đọc bao phủ)
    
    Returns:
    - Giá trị fitness của particle
    """
    # X = c / T: Tỷ lệ số item được bao phủ
    X = students_covered / total_students
    # Y = (N - n) / N: Tỷ lệ giảm của số đầu đọc được sử dụng
    Y = (total_readers - readers_used) / total_readers
    # Z = (T - e) / T: Tỷ lệ giảm của số điểm bị bao phủ bởi nhiều đầu đọc
    Z = (total_students - overlap_points) / total_students
    # Hàm fitness: 0.6 * X + 0.2 * Y + 0.2 * Z
    fitness_value = 0.6 * X + 0.2 * Y  + 0.2 * Z
    return fitness_value

class SSPSO:
    def __init__(self, num_particles, dim, max_iter):
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.readers = [Readers(dim) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(-1, 1, dim)
        self.global_best_value = float('inf')

    def optimize(self, STUDENTS):
        for _ in range(self.max_iter):
            for particle in self.readers:
                # Đánh giá hàm mục tiêu dựa trên độ phủ và trùng lặp
                fitness_value = fitness(calculate_covered_students(self.readers, STUDENTS), len(STUDENTS), calculate_overlap(self.readers, STUDENTS, RFID_RADIUS), self.num_particles, self.num_particles)
                
                # Cập nhật vị trí tốt nhất của từng hạt
                if fitness_value > particle.best_value:  # Tối đa hóa
                    particle.best_position = particle.position.copy()
                    particle.best_value = fitness_value

                # Cập nhật vị trí tốt nhất của toàn bộ quần thể
                if fitness_value > self.global_best_value:  # Tối đa hóa
                    self.global_best_position = particle.position.copy()
                    self.global_best_value = fitness_value      
        
        
            


readers =  [Readers(DIM) for _ in range(NUM_RFID_READERS)]
students = [Students(DIM) for _ in range(NUM_INDIVIDUALS)]
BieuDoReader(readers, students)
BieuDoStudents(readers, students)



