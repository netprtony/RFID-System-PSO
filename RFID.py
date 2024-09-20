import numpy as np
import matplotlib.pyplot as plt

grid_x, grid_y = 10, 6 # Kích thước của 1 lớp học
num_individuals = 50 # Sô lượng sinh viên
num_rfid_readers = 3  # Số lượng đầu đọc RFID
individuals = np.random.rand(num_individuals, 2) * [grid_x, grid_y]
num_iterations = 200  # Số vòng lặp
# Tạo 3 điểm ngẫu nhiên cho các đầu đọc RFID
#rfid_readers = np.random.rand(num_rfid_readers, 2) * [grid_x, grid_y]
rfid_radius = 3.69 # Bán kính vùng phủ sóng của đầu đọc

# Hàm để kiểm tra khoảng cách giữa hai điểm
def is_valid_distance(new_point, points, min_distance):
    for point in points:
        if np.linalg.norm(new_point - point) < min_distance:
            return False
    return True

# Tạo ngẫu nhiên vị trí ban đầu của các đầu đọc RFID với điều kiện khoảng cách
rfid_readers = np.zeros((num_rfid_readers, 2))
count = 0
while count < num_rfid_readers:
    new_reader = np.random.rand(2) * [grid_x, grid_y]
    if is_valid_distance(new_reader, rfid_readers[:count], rfid_radius):  # Kiểm tra khoảng cách với các đầu đọc đã thêm
        rfid_readers[count] = new_reader
        count += 1

rfid_readers = np.array(rfid_readers)

# Vận tốc ban đầu của các đầu đọc RFID
velocities = np.random.rand(num_rfid_readers, 2) * 0.1  # Tốc độ nhỏ


def show_chart(individuals, rfid_readers):
    # Hiển thị kết quả trên biểu đồ
    fig, ax = plt.subplots()

    # Vẽ các cá thể dưới dạng các điểm màu xanh
    ax.scatter(individuals[:, 0], individuals[:, 1], color='blue', label='Individuals')

    # Vẽ các điểm đầu đọc RFID dưới dạng các điểm màu đỏ
    ax.scatter(rfid_readers[:, 0], rfid_readers[:, 1], color='red', label='RFID Readers')

    # Vẽ vòng tròn vùng phủ sóng cho các đầu đọc RFID
    for (x, y) in rfid_readers:
        circle = plt.Circle((x, y), rfid_radius, color='red', fill=True, alpha=0.2, linestyle='--')
        ax.add_artist(circle)

    # Đặt nhãn trục và tiêu đề
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('RFID Reader Positions Optimized by SSPSO')

    # Thiết lập giới hạn của trục x và y
    ax.set_xlim(0, grid_x)  
    ax.set_ylim(0, grid_y)  

    # Thiết lập tỷ lệ của biểu đồ
    ax.set_aspect('equal', 'box')

    # Hiển thị chú thích (legend)
    #ax.legend()

    # Hiển thị biểu đồ
    plt.show()

# Hàm mục tiêu: Đếm số cá thể không được bao phủ bởi bất kỳ đầu đọc nào
def objective_function(rfid_readers):
    uncovered_count = 0
    for individual in individuals:
        covered = False
        for reader in rfid_readers:
            distance = np.linalg.norm(individual - reader)
            if distance <= rfid_radius:
                covered = True
                break
        if not covered:
            uncovered_count += 1
    return uncovered_count

# Lưu vị trí tốt nhất của các đầu đọc
pbest_positions = np.copy(rfid_readers)
pbest_scores = np.full(num_rfid_readers, np.inf)  # Số lượng cá thể không được bao phủ (càng thấp càng tốt)

gbest_position = rfid_readers[0]  # Khởi tạo với một giá trị hợp lệ từ vị trí đầu đọc ban đầu
gbest_score = np.inf 

# PSO - SSPSO
for iteration in range(num_iterations):
    for i in range(num_rfid_readers):
        # Đánh giá hàm mục tiêu cho từng cá thể
        current_score = objective_function(rfid_readers)

        # Cập nhật vị trí tốt nhất của cá thể
        if current_score < pbest_scores[i]:
            pbest_scores[i] = current_score
            pbest_positions[i] = rfid_readers[i]

        # Cập nhật vị trí tốt nhất của toàn bộ quần thể
        if current_score < gbest_score:
            gbest_score = current_score
            gbest_position = rfid_readers[i]

    # Kiểm tra nếu tất cả các cá thể đã được bao phủ
    if gbest_score == 0:
        print(f"Tìm thấy giải pháp tối ưu tại vòng lặp {iteration}")
        break

    # Cập nhật vận tốc và vị trí của các đầu đọc RFID
    for i in range(num_rfid_readers):
        # Cập nhật vận tốc
        inertia = 0.5  # Hệ số quán tính
        cognitive = 0.8  # Hệ số học hỏi cá nhân
        social = 0.9  # Hệ số học hỏi xã hội

        r1, r2 = np.random.rand(2)
        velocities[i] = (inertia * velocities[i] +
                         cognitive * r1 * (pbest_positions[i] - rfid_readers[i]) +
                         social * r2 * (gbest_position - rfid_readers[i]))

        # Cập nhật vị trí
        new_position = rfid_readers[i] + velocities[i]
        # Giới hạn vị trí trong vùng làm việc
        new_position[0] = np.clip(new_position[0], 0, grid_x)
        new_position[1] = np.clip(new_position[1], 0, grid_y)

        if is_valid_distance(new_position, np.delete(rfid_readers, i, axis=0), rfid_radius):
            # In ra vị trí trước và sau khi cập nhật
            print(f"RFID Reader {i+1} cập nhật vị trí từ {rfid_readers[i]} đến {new_position}")
            rfid_readers[i] = new_position
        show_chart(individuals, rfid_readers)

show_chart(individuals, rfid_readers)