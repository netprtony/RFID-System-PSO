import numpy as np
import time
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from classes import Readers, Tags, FireflyAlgorithm, ParticleSwarmOptimizationAlgorithm
from classes import  GRID_X, GRID_Y
from utils import calculate_covered_tags, calculate_interference_basic, fitness_function_basic, RFID_RADIUS
UPDATE_INTERVAL = 500
DIM = 2
EXCLUSION_FORCE = 0.2 # Hệ số lực đẩy
ATTRACTION_FORCE = 1  # Hệ số lực hút

# Định nghĩa hằng số cho lực
REPULSION_FORCE_COEF = 1.0  # Hệ số lực đẩy
ATTRACTION_FORCE_COEF = 0.5  # Hệ số lực hút
IDEAL_DISTANCE = 10.0 # Khoảng cách lý tưởng giữa các đầu đọc
GRID_SIZE = 1.6
def initialize_readers_with_kmeans(tags, num_readers):
    """Khởi tạo vị trí đầu đọc sử dụng thuật toán KMeans."""
    positions = np.array([tag.position for tag in tags])
    kmeans = KMeans(n_clusters=num_readers, random_state=42).fit(positions)
    return [Readers(position=center) for center in kmeans.cluster_centers_]
    
def create_grid(grid_size, grid_x, grid_y):
    x_points = np.arange(0, grid_x + grid_size, grid_size)
    y_points = np.arange(0, grid_y + grid_size, grid_size)
    grid_points = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1, 2)
    return grid_points

def snap_to_grid(position, grid_points):
    distances = np.linalg.norm(grid_points - position, axis=1)
    nearest_index = np.argmin(distances)
    return grid_points[nearest_index]

def Reader_GRID(readers, GRID_SIZE, GRID_X, GRID_Y):
    grid_points = create_grid(GRID_SIZE, GRID_X, GRID_Y)
    
    # Điều chỉnh vị trí các đầu đọc về mắt lưới gần nhất và giới hạn ở các biên của biểu đồ
    for reader in readers:
        snapped_position = snap_to_grid(reader.position, grid_points)
        
        # Giới hạn vị trí trong phạm vi của biểu đồ
        snapped_position[0] = max(0, min(snapped_position[0], GRID_X))
        snapped_position[1] = max(0, min(snapped_position[1], GRID_Y))
        
        reader.position = snapped_position
    
    return readers

def selection_mechanism(tags, initial_num_readers, COVER_THRESHOLD):
    """Hàm chọn đầu đọc dựa trên KMeans và điều chỉnh vị trí về mắt lưới."""
    readers = []  # Danh sách đầu đọc
    num_readers = initial_num_readers  # Số lượng đầu đọc ban đầu
    #Tạo lưới
    #grid_points = create_grid(GRID_SIZE, grid_x, grid_y)
    
    while True:
        # Khởi tạo các đầu đọc với vị trí cụm từ KMeans
        kmeans_readers = initialize_readers_with_kmeans(tags, num_readers)

        # #Điều chỉnh vị trí các đầu đọc về mắt lưới gần nhất
        # for reader in kmeans_readers:
        #     reader.position = snap_to_grid(reader.position, grid_points)

        # Đặt trạng thái bao phủ của tất cả các thẻ
        for tag in tags:
            tag.covered = False
            # Kiểm tra nếu thẻ nằm trong vùng phủ sóng của bất kỳ đầu đọc nào
            for reader in kmeans_readers:
                if np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS:
                    tag.covered = True
                    break

        # Tính tỷ lệ thẻ được bao phủ
        coverage_ratio = calculate_covered_tags(kmeans_readers, tags, RFID_RADIUS) / 100
        print(f"Độ bao phủ: {coverage_ratio:.2%}")
        #BieuDoReader(kmeans_readers, tags)
        # Nếu tỷ lệ bao phủ đạt yêu cầu, thoát khỏi vòng lặp
        if coverage_ratio >= COVER_THRESHOLD:
            readers = kmeans_readers
            break

        # Nếu không đạt, tăng số lượng đầu đọc và lặp lại
        num_readers += 1
    print(f"Số lượng đầu đọc được chọn: {num_readers}")
    return readers  # Trả về danh sách đầu đọc đã được chọn



def adjust_readers_location_by_virtual_force(readers, tags, max_no_change_iterations=5):
    no_change_iterations = 0
    best_fitness = -float('inf')
    best_positions = [reader.position.copy() for reader in readers]

    while no_change_iterations < max_no_change_iterations:
        for reader in readers:
            # Lực tổng hợp được khởi tạo
            total_exclusion_force = np.array([0.0, 0.0])
            total_attraction_force = np.array([0.0, 0.0])

            # 1. Lực đẩy (Exclusion Operator) (8) (9)
            for other_reader in readers:
                if other_reader != reader:
                    distance = np.linalg.norm(reader.position - other_reader.position)
                    if distance < 2 * RFID_RADIUS:  # Kiểm tra xem hai đầu đọc có chồng lấn không
                        # Tính toán lực đẩy
                        force_magnitude = EXCLUSION_FORCE * (2 * RFID_RADIUS - distance)
                        direction = (reader.position - other_reader.position) / distance
                        total_exclusion_force += force_magnitude * direction

            # 2. Lực hút (Attraction Operator) (10) (11)
            for tag in tags:
                if not tag.covered:  # Nếu thẻ chưa được bao phủ
                    distance = np.linalg.norm(reader.position - tag.position)
                    if distance <= RFID_RADIUS:  # Kiểm tra xem thẻ có nằm trong phạm vi hấp dẫn không
                        # Tính toán lực hút
                        force_magnitude = ATTRACTION_FORCE * (RFID_RADIUS - distance)
                        direction = (tag.position - reader.position) / distance
                        total_attraction_force += force_magnitude * direction

            # 3. Cập nhật vị trí đầu đọc dựa trên lực tổng hợp(12)(13)
            total_force = total_exclusion_force + total_attraction_force
            reader.position += total_force
        
            # Giới hạn vị trí trong không gian làm việc
            reader.position[0] = np.clip(reader.position[0], 0, GRID_X )
            reader.position[1] = np.clip(reader.position[1], 0, GRID_Y )

        # Tính tỷ lệ thẻ được bao phủ
        # Tính toán giá trị fitness
        COV = calculate_covered_tags(readers, tags)
        ITF = calculate_interference_basic(readers, tags, RFID_RADIUS)
        fitness = fitness_function_basic(COV, ITF, tags, 0.8, 0.2)
        print(Fore.LIGHTYELLOW_EX + f"Fitness: {fitness}")
        # Kiểm tra nếu giá trị fitness tốt hơn
        if fitness > best_fitness:
            best_fitness = fitness
            best_positions = [reader.position.copy() for reader in readers]
            no_change_iterations = 0
        else:
            no_change_iterations += 1

        # Khôi phục vị trí tốt nhất
        for reader, best_position in zip(readers, best_positions):
            reader.position = best_position
    print(f"Final fitness: {best_fitness}")
    return readers
def BieuDoReader(readers, tags, title, GRID_SIZE):
    """
    Vẽ biểu đồ vị trí các đầu đọc và các thẻ với mặt lưới.
 
    Parameters:
    - readers: Danh sách các đối tượng reader.
    - tags: Danh sách các đối tượng tag.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Hiển thị mặt lưới
    ax.set_xticks(np.arange(0, GRID_X + 1, GRID_SIZE))
    ax.set_yticks(np.arange(0, GRID_Y + 1, GRID_SIZE))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Vẽ các thẻ
    tag_positions = np.array([tag.position for tag in tags])
    tag_colors = ['red' if not any(np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS for reader in readers if reader.active) else 'green' for tag in tags]
    ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color=tag_colors, label='Tags', s=20, marker='x')

    # Vẽ các đầu đọc
    active_reader_positions = np.array([reader.position for reader in readers if reader.active])
    ax.scatter(active_reader_positions[:, 0], active_reader_positions[:, 1], color='blue', label='Readers', marker='^')

    # Vẽ các vòng tròn phạm vi phủ sóng
    for reader in readers:
        if reader.active:
            circle = plt.Circle(reader.position, RFID_RADIUS, color='black', fill=False, linestyle='-', linewidth=1, alpha=0.5)
            ax.add_artist(circle)

    # Thêm thông tin độ phủ sóng, độ nhiễu và số lượng đầu đọc
    coverage_tag = calculate_covered_tags(readers, tags, RFID_RADIUS)/ 100 * len(tags)  # Phải định nghĩa hàm calculate_covered_tags
    interference = calculate_interference_basic(readers, tags, RFID_RADIUS)  # Phải định nghĩa hàm calculate_interference_basic
    active_reader_count = sum(reader.active for reader in readers)

    # Di chuyển các thông tin lên đầu biểu đồ
    fig.text(0.3, 0.92, f"Có {len(tags)} thẻ được bao phủ {coverage_tag:.0f} thẻ", fontsize=12, color="black", ha='left', va='top')
    #fig.text(0.5, 0.92, f"Độ nhiễu: {interference:.2f}%", fontsize=12, color="orange", ha='left', va='top')
    fig.text(0.75, 0.92, f"{active_reader_count} đầu đọc", fontsize=12, color="blue", ha='left', va='top')

    # Đặt tiêu đề cho biểu đồ
    fig.suptitle(title, fontsize=14, ha='left', va='top', fontweight='bold', x=0.01, y=0.99)
    ax.legend(loc='upper right')  # Đặt chú thích ở góc trên bên phải bên trong biểu đồ
    plt.show()
def BieuDoReaderTongHop(readers_pso, readers_fa, tags, GRID_SIZE, GRID_X, GRID_Y, RFID_RADIUS):
    """
    Vẽ biểu đồ vị trí các đầu đọc và các thẻ với mặt lưới cho hai thuật toán.

    Parameters:
    - readers_pso: Danh sách các đối tượng reader của thuật toán PSO.
    - readers_fa: Danh sách các đối tượng reader của thuật toán FA.
    - tags: Danh sách các đối tượng tag.
    - title_pso: Tiêu đề của biểu đồ PSO.
    - title_fa: Tiêu đề của biểu đồ FA.
    - GRID_SIZE: Kích thước của lưới.
    - GRID_X: Chiều rộng của biểu đồ.
    - GRID_Y: Chiều cao của biểu đồ.
    - RFID_RADIUS: Bán kính phủ sóng của đầu đọc.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    def plot_readers(ax, readers, title):
        ax.set_xlim(0, GRID_X)
        ax.set_ylim(0, GRID_Y)
        ax.set_aspect('equal', 'box')

        # Hiển thị mặt lưới
        ax.set_xticks(np.arange(0, GRID_X + 1, GRID_SIZE))
        ax.set_yticks(np.arange(0, GRID_Y + 1, GRID_SIZE))
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Vẽ các thẻ
        tag_positions = np.array([tag.position for tag in tags])
        tag_colors = ['red' if not any(np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS for reader in readers if reader.active) else 'green' for tag in tags]
        ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color=tag_colors, label='Tags', s=20, marker='x')

        # Vẽ các đầu đọc
        active_reader_positions = np.array([reader.position for reader in readers if reader.active])
        ax.scatter(active_reader_positions[:, 0], active_reader_positions[:, 1], color='blue', label='Readers', marker='^')

        # Vẽ các vòng tròn phạm vi phủ sóng
        for reader in readers:
            if reader.active:
                circle = plt.Circle(reader.position, RFID_RADIUS, color='black', fill=False, linestyle='-', linewidth=1, alpha=0.5)
                ax.add_artist(circle)

        # Thêm thông tin độ phủ sóng, độ nhiễu và số lượng đầu đọc
        coverage_tag = calculate_covered_tags(readers, tags, RFID_RADIUS) / 100 * len(tags)  # Phải định nghĩa hàm calculate_covered_tags
        interference = calculate_interference_basic(readers, tags, RFID_RADIUS)  # Phải định nghĩa hàm calculate_interference_basic
        active_reader_count = sum(reader.active for reader in readers)

        # Di chuyển các thông tin lên đầu biểu đồ
        # fig.text(0.3, 0.92, fig_text_coverage.format(len(tags), coverage_tag), fontsize=12, color="black", ha='left', va='top')
        # fig.text(0.5, 0.92, fig_text_readers.format(active_reader_count), fontsize=12, color="blue", ha='left', va='top')
        # fig.text(0.75, 0.92, fig_text_interference.format(interference), fontsize=12, color="red", ha='left', va='top')
        # Đặt tiêu đề cho biểu đồ
        ax.set_title(title)
        ax.legend(loc='upper right')  # Đặt chú thích ở góc trên bên phải bên trong biểu đồ

    coverage_tagPSO = int(calculate_covered_tags(readers_pso, tags, RFID_RADIUS) / 100 * len(tags))  # Phải định nghĩa hàm calculate_covered_tags
    interferencePSO = calculate_interference_basic(readers_pso, tags, RFID_RADIUS)  # Phải định nghĩa hàm calculate_interference_basic
    active_reader_countPSO = sum(reader.active for reader in readers_pso)

    coverage_tagFA = int(calculate_covered_tags(readers_fa, tags, RFID_RADIUS) / 100 * len(tags))  # Phải định nghĩa hàm calculate_covered_tags
    interferenceFA= calculate_interference_basic(readers_fa, tags, RFID_RADIUS)  # Phải định nghĩa hàm calculate_interference_basic
    active_reader_countFA = sum(reader.active for reader in readers_fa)
    # Vẽ biểu đồ cho PSO
    plot_readers(ax1, readers_pso, f"PSO: {active_reader_countPSO} đầu đọc bao phủ {coverage_tagPSO}/{len(tags)} thẻ, nhiễu: {interferencePSO:.2f}%")
    
    # Vẽ biểu đồ cho FA
    plot_readers(ax2, readers_fa, f"FA: {active_reader_countFA} đầu đọc bao phủ {coverage_tagFA}/{len(tags)} thẻ, nhiễu: {interferenceFA:.2f}%""")

    plt.show()
def BieuDoSoSanh(value1, value2, xlabel, ylabel, title):
    # Tách dữ liệu từ tracking
    v1_x = [item[0] for item in value1]
    v1_y = [item[1] for item in value1]
    v2_x = [item[0] for item in value2]
    v2_y = [item[1] for item in value2]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(v1_x, v1_y, marker='D', label='FA', color='blue', linewidth=2)  # HGA với marker là diamond
    plt.plot(v2_x, v2_y, marker='s', label='PSO', color='orange', linewidth=2)  # HPSO với marker là square
    
    # Labels and legend
    plt.title(f"{title}", fontsize=18)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.ylim(70, 100)  # Giới hạn trục y
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)  # Lưới nhẹ
    
    # Show the plot
    plt.show()


def PSO_Algorithm(readers, tags, GRID_X, GRID_Y, GRID_SIZE):
    reader_init = len(readers)
    start_time = time.time()
    sspso = ParticleSwarmOptimizationAlgorithm(len(readers), DIM, 100, readers)
    sspso.readers, itr_stop, bestFitness = sspso.optimize(tags, RFID_RADIUS)
    #BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi tối ưu hội tụ", GRID_SIZE)
    sspso.readers = adjust_readers_location_by_virtual_force(sspso.readers, tags)
    #BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi tối ưu hóa bằng lực ảo", GRID_SIZE)
    sspso.readers = Reader_GRID(sspso.readers, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    #BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi đưa vị trí về mắt lưới", GRID_SIZE)     
    cov_before = calculate_covered_tags(readers, tags)  
    ift_before = calculate_interference_basic(readers, tags)
    #sspso.readers = Redundant_Reader_Elimination(sspso.readers, tags)
    #BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi loại bỏ đầu đọc dư thừa", GRID_SIZE)
    end_time = time.time()
    print(f"Optimization stopped after {itr_stop} iterations.")
    print(f"Optimization time: {end_time - start_time:.2f} seconds.")
    print(f"Số lượng đầu đọc ban đầu: {reader_init}")
    print(f"Số lượng đầu đọc cuối cùng: {len(sspso.readers)}")
    print(Fore.RED + f"Độ bao phủ khi chưa loại bỏ dư thừa {cov_before:.2f}")
    print(Fore.RED + f"Độ nhiễu khi chưa loại bỏ dư thừa: {ift_before:.2f}")
    print(Fore.GREEN +"Độ bao phủ cuối cùng: {:.2f}%".format(calculate_covered_tags(sspso.readers, tags)))
    print(Fore.GREEN +"Nhiễu cuối cùng: {:.2f}".format(calculate_interference_basic(sspso.readers, tags)))
    print(F"Best fitness: {bestFitness:.2f}")
    #BieuDoReader(sspso.readers, tags, "PSO", GRID_SIZE)
    return sspso.readers
    

def Redundant_Reader_Elimination(readers, tags, coverage_threshold=1, interference_threshold = 10):
    """
    Loại bỏ các đầu đọc dư thừa dựa trên ba tiêu chí:
    1. Giảm ít hơn 1% tỷ lệ bao phủ và tổng tỷ lệ bao phủ không giảm dưới 90%.
    2. Giảm nhiễu hơn 10%.
    3. Giảm giá trị hàm fitness ít hơn 1%.

    Parameters:
    - readers: Danh sách các đối tượng reader
    - tags: Danh sách các đối tượng tag
    - coverage_threshold: Ngưỡng tỷ lệ bao phủ tối thiểu (mặc định là 90%)
    - interference_threshold: Ngưỡng giảm nhiễu tối thiểu (mặc định là 10%)
    - fitness_threshold: Ngưỡng giảm giá trị hàm fitness tối thiểu (mặc định là 1%)
    - w1, w2: Trọng số cho các thành phần độ phủ và nhiễu

    Returns:
    - readers: Danh sách các đối tượng reader sau khi loại bỏ các đầu đọc dư thừa
    """
    initial_coverage = calculate_covered_tags(readers, tags)
    initial_interference = calculate_interference_basic(readers, tags) 
    #initial_fitness = fitness_function_basic(initial_coverage, initial_interference, tags, w1, w2)
    readers_to_remove = []

    for i, reader in enumerate(readers):
        if reader.active:
            # Tạm thời tắt đầu đọc
            reader.active = False

            # Tính toán lại các giá trị
            new_coverage = calculate_covered_tags(readers, tags)
            new_interference = calculate_interference_basic(readers, tags)
            #new_fitness = fitness_function_basic(new_coverage, new_interference, tags, w1, w2)
           
            # Kiểm tra các tiêu chí
            coverage_reduction = initial_coverage - new_coverage
            interference_reduction = initial_interference - new_interference
            #fitness_reduction = initial_fitness - new_fitness

            if coverage_reduction < coverage_threshold and new_coverage >= 80 and interference_reduction < interference_threshold :#and
                #fitness_reduction < fitness_threshold):
                print(Fore.GREEN + f"Đã loại bỏ thành công đầu đọc {i}.")
                readers_to_remove.append(reader)
            else:
                # Khôi phục đầu đọc nếu không thỏa mãn các tiêu chí
                reader.active = True

    for reader in readers_to_remove:
        readers.remove(reader)

    return readers
def tracking_PSO_time(sspso, tags, GRID_SIZE):
    start_time = time.time()
    sspso.optimize(tags)
    sspso.readers = adjust_readers_location_by_virtual_force(sspso.readers, tags)
    sspso.readers = Reader_GRID(sspso.readers, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    end_time = time.time()
    return int(end_time - start_time)

def tracking_PSO_COV(sspso, tags, GRID_SIZE):
    sspso.optimize(tags)
    sspso.readers = adjust_readers_location_by_virtual_force(sspso.readers, tags)
    sspso.readers = Reader_GRID(sspso.readers, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    COV = calculate_covered_tags(sspso.readers, tags)
    return COV
def tracking_PSO_IFT(sspso, tags, GRID_SIZE):
    sspso.optimize(tags)
    sspso.readers = adjust_readers_location_by_virtual_force(sspso.readers, tags)
    sspso.readers = Reader_GRID(sspso.readers, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    ITF = calculate_interference_basic(sspso.readers, tags)
    return ITF
def tracking_PSO_Fitness(sspso, tags, GRID_SIZE):
    sspso.optimize(tags)
    sspso.readers = adjust_readers_location_by_virtual_force(sspso.readers, tags)
    sspso.readers = Reader_GRID(sspso.readers, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    fitness = fitness_function_basic(calculate_covered_tags(sspso.readers, tags), calculate_interference_basic(sspso.readers, tags), tags, 0.8, 0.2)
    return fitness

def tracking_FA_time(reader_FA, tags, GRID_SIZE):
    start_time = time.time()
    FA_Algorithm(reader_FA, tags, 50 , 50, GRID_SIZE)
    reader_FA = adjust_readers_location_by_virtual_force(reader_FA, tags)
    reader_FA = Reader_GRID(reader_FA, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    end_time = time.time()
    return int(end_time - start_time)

def tracking_FA_COV(reader_FA, tags, GRID_SIZE):
    FA_Algorithm(reader_FA, tags, 50 , 50, GRID_SIZE)
    reader_FA = adjust_readers_location_by_virtual_force(reader_FA, tags)
    reader_FA = Reader_GRID(reader_FA, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    COV = calculate_covered_tags(reader_FA, tags)
    return COV
def tracking_FA_IFT(reader_FA, tags, GRID_SIZE):
    FA_Algorithm(reader_FA, tags, 50 , 50, GRID_SIZE)
    reader_FA = adjust_readers_location_by_virtual_force(reader_FA, tags)
    reader_FA = Reader_GRID(reader_FA, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    ITF = calculate_interference_basic(reader_FA, tags)
    return ITF
def tracking_FA_Fitness(reader_FA, tags, GRID_SIZE):
    FA_Algorithm(reader_FA, tags, 50 , 50, GRID_SIZE)
    reader_FA = adjust_readers_location_by_virtual_force(reader_FA, tags)
    reader_FA = Reader_GRID(reader_FA, GRID_SIZE, GRID_X=GRID_X, GRID_Y=GRID_Y) 
    fitness = fitness_function_basic(calculate_covered_tags(reader_FA, tags), calculate_interference_basic(reader_FA, tags), tags, 0.8, 0.2)
    return fitness

def FA_Algorithm(readers, tags, GRID_X, GRID_Y, GRID_SIZE):
    reader_init = len(readers)
    start_time = time.time()
    listFA_position = [reader.position for reader in readers]
    fa = FireflyAlgorithm(listFA_position, len(readers))
    firefilies, itr_stop, bestFitness = fa.optimize(tags)
    firefilies = adjust_readers_location_by_virtual_force(firefilies, tags)
    firefilies = Reader_GRID(firefilies, GRID_SIZE, GRID_X, GRID_Y) 
    cov_before = calculate_covered_tags(firefilies, tags)
    ift_before = calculate_interference_basic(firefilies, tags)
    #firefilies = Redundant_Reader_Elimination(firefilies, tags)
    end_time = time.time()
    print(f"Optimization stopped after {itr_stop} iterations.")
    print(f"Optimization time: {end_time - start_time:.2f} seconds.")
    print(f"Số lượng đầu đọc ban đầu: {reader_init}")
    print(f"Số lượng đầu đọc cuối cùng: {len(firefilies)}")
    print(f"Độ bao phủ khi chưa loại bỏ dư thừa {cov_before:.2f}")
    print(f"Độ nhiễu khi chưa loại bỏ dư thừa: {ift_before:.2f}")
    print("Độ bao phủ cuối cùng: {:.2f}%".format(calculate_covered_tags(firefilies, tags)))
    print("Nhiễu cuối cùng: {:.2f}".format(calculate_interference_basic(firefilies, tags)))
    print(F"Best fitness: {bestFitness:.2f}")
    #BieuDoReader(firefilies, tags, "Firefly Algorithm", GRID_SIZE)

tag_positions = np.array([
    [41.233776269385814, 32.43297604049532],
    [18.134158949016477, 21.39321904604326],
    [25.806499145464578, 23.742477292369475],
    [16.92453383081944, 12.882239417168067],
    [11.085518229933077, 47.92530160341377],
    [33.62880126779815, 36.58824293344932],
    [27.83995975828986, 2.172176337445358],
    [30.196144649497032, 34.25784054881891],
    [38.485774359439375, 7.255923708318757],
    [0.2332538177498844, 41.967478256746055],
    [40.02788355633607, 18.782873747040068],
    [2.6561556039903733, 35.07578104062479],
    [35.89427245548806, 41.11376670484893],
    [25.33666921405528, 21.30677434316536],
    [41.352497908087905, 6.408561834632015],
    [27.38682489637118, 10.241050404351576],
    [49.17816192079861, 36.605394156387014],
    [46.204470081624336, 34.15121453491367],
    [24.386888309818627, 36.40097005009991],
    [27.06792602390954, 38.264894939508544],
    [39.96230821044814, 27.417722368906],
    [39.75631978915955, 13.020021993738734],
    [46.46174405999468, 32.40569549141],
    [13.141000474396048, 32.866214216262875],
    [44.19756868834529, 10.259943853157981],
    [23.007241982645205, 43.58454505126199],
    [44.42430959059712, 22.066391688055575],
    [7.261803465506778, 9.52941055083802],
    [45.52741377657645, 2.7125021870734676],
    [37.911073510215246, 12.479500931248571],
    [3.1870293833014482, 33.67993287430156],
    [9.593273831225863, 15.92281094474136],
    [28.923280189473914, 6.369480774845293],
    [15.198729718674182, 42.710586337610934],
    [38.47499081891906, 42.108055228411956],
    [12.901924330963272, 34.006664370836326],
    [8.562643967703071, 36.76881871758306],
    [3.6756879898924955, 17.709337541515374],
    [23.78641178684966, 33.14624666491426],
    [43.18854569446603, 12.305055699310607],
    [43.379147728742375, 38.171219461251134],
    [30.676518578174196, 2.632965022230793],
    [1.4512670799982197, 37.35228877310503],
    [33.988800946149496, 1.9614809920822096],
    [11.890875622233594, 38.88903925086211],
    [3.306182454339196, 22.10075700932636],
    [5.609052251975166, 22.843056052963874],
    [37.32073978322936, 8.442520349573329],
    [14.81333130879806, 49.476589301963095],
    [48.03730594705691, 20.919209840781345],
    [45.751016269274686, 45.796626794915426],
    [2.987594588927606, 0.8833147975999711],
    [13.987998304669025, 1.424081003877653],
    [34.506594273779115, 46.05750315469492],
    [43.478058747945035, 44.70176951193441],
    [7.837539696013495, 28.51121617675713],
    [10.057931420666733, 0.5165625566148413],
    [22.06524326382697, 2.4514353176042505],
    [15.44681349738618, 38.27132733829473],
    [6.041201071639746, 8.550014747101548],
    [19.875440502026322, 37.54019136040815],
    [0.4203185173335955, 0.7135369974078931],
    [46.85096160670164, 42.22222800350393],
    [39.893244938417254, 44.293356640036684],
    [41.054988663246824, 30.599135620364574],
    [14.87506847180638, 38.63119328131066],
    [25.255536529105015, 32.3063706248102],
    [24.018202260513455, 44.379873630487495],
    [43.71893831270181, 18.96573702287876],
    [22.63564953518753, 22.439466874435094],
    [18.45900825365952, 34.706869779976614],
    [40.616307389267305, 32.086980910344884],
    [2.1051311901045366, 13.381941277513004],
    [27.721892525467663, 7.997840870266898],
    [33.37330765769609, 21.47597522969632],
    [42.45232949135003, 5.127726865023346],
    [43.96963686043851, 5.020166563171963],
    [35.74046203519532, 2.9860210930914457],
    [16.23180233924994, 17.493695313670926],
    [35.74258981033465, 1.3851312459883025],
    [46.10108033857605, 23.415964004142882],
    [27.021092768175663, 3.5050972820440784],
    [13.03396925368277, 20.146811371063063],
    [4.218202296704298, 21.078248484248807],
    [22.546368318971993, 48.360737869554406],
    [8.55308364648158, 48.60925788894045],
    [32.80251965653373, 44.266564476055166],
    [0.46847563101014567, 2.052162564186805],
    [39.444228206005825, 22.542577995435792],
    [3.3441575700869697, 34.459990462400825],
    [1.463481972858166, 42.42007303417052],
    [36.17728078581088, 8.929335694839214],
    [21.200270710726155, 5.503068304432407],
    [31.253503677643877, 16.385833096661408],
    [38.335967828587194, 47.29817029378901],
    [29.076519173033144, 37.55032833189802],
    [3.2089456413343207, 23.218276525371678],
    [1.1125832263569224, 32.66483839825616],
    [26.04376885726845, 46.4627291198559],
    [36.20945047697953, 3.2342937394312763],
    [1.14592718399103, 28.32262201578921],
    [14.339879922180938, 47.59072813724949],
    [7.793253908294773, 35.45629674421],
    [27.6320436107258, 39.14253536095928],
    [3.636879220785877, 26.86589719409061],
    [3.7906804280024406, 8.272659192477333],
    [30.877473811069283, 35.2118307696625],
    [25.197881943836542, 9.987110968898943],
    [17.223229059637944, 34.70219494962137],
    [26.694990095069006, 38.54254565679272],
    [38.92283897049581, 34.71274607114851],
    [6.183292526910089, 3.7352163985448272],
    [34.37658813320914, 5.078564903983185],
    [48.12257384758219, 40.60272701744913],
    [3.232342570820851, 12.705885161094349],
    [1.9620327426582373, 33.735703028311455],
    [14.97050831427652, 23.640314339699536],
    [9.114737975059223, 17.390976997351586],
    [32.384591102785116, 34.289084533754895],
    [11.606898189895997, 34.95106472365911],
    [34.677134696707306, 2.8584857735218403],
    [13.958081884387962, 11.105676169534123],
    [21.25816553530194, 41.037469517245384],
    [14.855184899126145, 36.779324747672305],
    [39.3887727455126, 44.44138945997171],
    [22.119254162377207, 22.89040903113636],
    [3.8153242637302243, 0.242261365463331],
    [30.4285805787749, 38.940134764532374],
    [47.43005729598449, 6.082193580732204],
    [42.310568649532534, 1.4596084941479237],
    [28.118201791167298, 36.43671204648016],
    [2.085424514076978, 40.91315376521487],
    [14.512962553958037, 36.405351490512125],
    [39.50270235123693, 44.27060615372927],
    [18.54825821987639, 47.65350034823668],
    [20.400047103284315, 1.226456167043971],
    [11.08674140205952, 20.65663741938286],
    [28.505758487470246, 43.84303223957019],
    [15.283914562856793, 48.54333884513467],
    [27.92120460469998, 33.35857958760912],
    [35.72297723633045, 18.432667489141124],
    [42.418065545139484, 22.821294002812824],
    [1.317550446893756, 9.005747469108572],
    [32.45389516208302, 6.41835616473338],
    [33.06850347435098, 32.156836701572075],
    [13.860429219043402, 40.45401541177412],
    [12.313613445191862, 12.20700244507521],
    [20.61051056320481, 41.88522428495773],
    [38.059849582327956, 45.81958098877747],
    [48.93125369297924, 0.011797012606801216],
    [16.24387693081198, 4.580020209638147],
    [36.62567776350708, 1.5557748049424058],
    [25.559693552292085, 43.43957478375275],
    [25.92488191649267, 14.95592793486144],
    [5.744777523840167, 28.257082150949998],
    [25.10806424740894, 20.95628537047568],
    [6.153684900594502, 24.838480522573757],
    [29.33071187718112, 37.999636816218626],
    [30.00500420313551, 10.544694134008342],
    [27.32948772016357, 18.61085615668603],
    [47.15767004035392, 32.311564536798194],
    [32.94816144257686, 4.304391060027407],
    [43.57412769193082, 25.989560840322344],
    [20.175678516164087, 18.955882680394026],
    [29.217155152976858, 34.67173670492842],
    [30.701423372236743, 4.557425492598954],
    [1.3099698840834173, 39.122000319438044],
    [23.26972135185536, 12.34528741440561],
    [44.5778638855445, 3.526182490957125],
    [24.56003666610046, 27.882402750819573],
    [30.48504080262574, 47.019944464622945],
    [47.274574230961, 21.80508459976004],
    [23.308357624252096, 14.07136907860398],
    [15.571409914190898, 32.60333938005806],
    [9.734818068183143, 33.64503030577906],
    [24.962003422006905, 28.551969632275227],
    [32.9101712277826, 22.8052290345437],
    [32.46863932604164, 23.66655128915611],
    [16.17287944066575, 16.14646941650697],
    [28.068560345546928, 20.401360799122298],
    [3.998155577333068, 6.006465331116628],
    [6.604661647486565, 27.95429119875511],
    [32.99019373390264, 15.206371165455646],
    [24.032773117346647, 42.99210179448282],
    [18.67444460447502, 46.322909390223174],
    [23.00102373025802, 24.882371482388514],
    [46.47874057474173, 27.393932997204622],
    [43.00241691001769, 7.54780156417989],
    [28.91854499300991, 13.751437856892606],
    [36.306678930868124, 31.40652855321201],
    [38.19695970278052, 14.403364963058547],
    [47.894834057376514, 30.403269953648604],
    [9.381552273065441, 37.620130593441665],
    [42.82184093820356, 36.628465180211656],
    [7.978485876455127, 48.61455732878908],
    [23.295254365901354, 8.173561216718312],
    [34.75323474094954, 29.056884901517634],
    [29.441237326345853, 38.18701867185361],
    [20.59971374822088, 19.501141868590917],
    [37.51649543792172, 15.38756481865492],
    [18.333241340721656, 30.93627243987464],
    [31.529549000834102, 30.05842446466993],
    [1.786644806507065, 35.26711477701094],
    [14.935006728430567, 25.1570759684695],
    [17.374695406064607, 43.6836226712068],
    [7.8608944269784375, 8.23012051547402],
    [21.868127763455774, 14.999025475547889],
    [41.03607825276095, 24.15509889382922],
    [27.396639941462663, 26.483571537375795],
    [7.834449678578453, 4.871466614038599],
    [16.15367694537072, 3.091496080333256],
    [35.91040966451211, 3.2245040463314667],
    [44.13784973709758, 42.60995164763725],
    [6.279827075399308, 9.25296414603145],
    [32.002602860714944, 43.63030631403957],
    [21.778935933084405, 5.244818455344202],
    [28.001690646575845, 29.29211180952222],
    [22.466121467389133, 17.32487460565536],
    [21.51112112561515, 20.884906263278353],
    [42.82820347519763, 33.56721796376093],
    [26.596207395600498, 40.769149928657825],
    [18.19019726866229, 41.239883127162294],
    [0.5602261307829981, 0.8072582647239646],
    [12.373212350411722, 1.5291625698953704],
    [28.13342538767698, 48.50091356442614],
    [41.31421510298045, 10.94274108908092],
    [48.64700526802067, 33.05572960988972],
    [46.63188266060877, 8.467710999884487],
    [3.696427583502082, 1.4236074926357756],
    [16.85315969170256, 13.342708748756449],
    [31.053915908515506, 38.584163341057234],
    [42.00148988998768, 3.5107488103761133],
    [3.072545438315688, 23.173090465026114],
    [4.734440140055479, 3.5357046327900763],
    [23.199943941998253, 38.202021792614985],
    [45.97293807385172, 43.386851197989444],
    [39.26338670097153, 13.053990564917301],
    [2.495982381359446, 29.99301777175108],
    [48.96583553738058, 15.529826814921838],
    [35.83484857347063, 44.404825225594024],
    [39.16885785272141, 21.50504519537053],
    [45.60829604540731, 15.30755386657095],
    [41.051304111184386, 14.116319212072636],
    [27.840374175528765, 17.49740841544781],
    [27.89294825192125, 32.08841992861935],
    [8.898576685187386, 21.73091657156113],
    [6.142454516164725, 30.37394548459324],
    [9.855032769285128, 29.023868246184247],
    [7.613205902552872, 9.148754867923149],
    [4.74677102256138, 46.16599427644565],
    [1.122028881700099, 26.12986796247951],
    [24.806747514796022, 46.27726729254971],
    [29.07628393602158, 19.016955727179386],
    [7.594205603300336, 10.226855963159153],
    [35.3534152472159, 6.995536980071853],
    [49.9975002886124, 39.14361784373658],
    [0.8204481952216292, 47.505274211470564],
    [35.18013547976632, 30.7181718032231],
    [31.78512665656934, 14.395869156454733],
    [22.908248849896374, 16.3376268788406],
    [48.92614769956577, 6.232290879895869],
    [7.086765214341229, 6.529663617492099],
    [16.249547285410443, 20.065710690234624],
    [20.72903982172859, 29.12315382030966],
    [12.969651446906044, 7.177822111962517],
    [48.878607519367314, 46.99298012617103],
    [49.945439242556944, 32.625624091879494],
    [11.965183483947762, 1.2110600301448615],
    [44.17232266151608, 19.469133406456507],
    [3.1964065511579207, 28.248916806486392],
    [1.245030522741486, 32.06897121364863],
    [33.2515091032473, 45.0097684811741],
    [42.59435399210897, 49.36574933013571],
    [29.266948975385848, 44.380020936328066],
    [11.012352286988442, 15.197893326718798],
    [32.16418349311685, 38.38020152176498],
    [45.93021627186313, 42.95738976166112],
    [33.69590717447787, 39.16815731391256],
    [0.4132472302606771, 34.60664935203418],
    [24.486811999260883, 29.4462499325619],
    [4.334678840393313, 39.6976814694583],
    [7.335685518914764, 48.272946904518165],
    [7.051525954212212, 44.492524375974234],
    [22.13409140359698, 42.92664829141402],
    [42.82517789740801, 31.071979636354236],
    [40.8320455612717, 3.9919377652597676],
    [0.24102463514187877, 28.720229545869323],
    [12.473044256386363, 10.084782322298663],
    [36.7312411458724, 6.529124335838066],
    [1.3299112270559077, 43.366840769806984],
    [31.92108100715332, 31.793544498837083],
    [2.4733683395843142, 3.860193482955454],
    [10.999364827540624, 4.548935401769327],
    [15.227629795576952, 30.444287127639],
    [47.94652288162691, 35.746023338844005],
    [3.0176126410315804, 42.72784984617194],
    [12.336456612643198, 10.492709767247177],
    [45.441094497521476, 36.66437020166131],
    [26.489779166400833, 17.15845089265382],
    [3.622177950042038, 23.51042223235047],
    [21.80874510018198, 27.614322782460732],
    [45.47261904822425, 39.19340487643464],
    [9.252511161205984, 2.222075550473929],
    [11.6371321778083, 2.659020655500771],
    [47.94253773479018, 0.09550301269062866],
    [49.549078097708986, 47.46674026873297],
    [43.24579641206451, 28.315726371172573],
    [22.22521390738915, 45.34327404327193],
    [30.742512930451355, 27.66254244916509],
    [22.43032993853288, 30.11936756110559],
    [36.759947936055525, 1.3768470903151586],
    [8.418811908996926, 21.229997687835482],
    [34.52937715814906, 8.138877471210131],
    [34.4997994974866, 10.819544605262898],
    [40.85229236549099, 21.132501217641792],
    [3.6325935607110225, 3.572313054020426],
    [32.90954508442209, 4.316325490375933],
    [28.805173948865686, 41.99979230671481],
    [20.11534507811787, 15.132738533830075],
    [34.8447867390889, 20.27394416439047],
    [28.57339889588418, 41.2505813064307],
    [8.57535274660865, 43.311323255061055],
    [21.285924710732633, 34.267782921931214],
    [40.81434418462044, 10.329731838059336],
    [0.17978226976900213, 46.51462165398543],
    [32.84724371527961, 23.446282962456333],
    [22.592667429908975, 35.631937080284764],
    [31.28087158886972, 3.0984185043113954],
    [21.035711189889163, 19.364186631117292],
    [25.482468405156837, 44.50034575372134],
    [34.43365091915253, 24.47935001349325],
    [14.396649435374115, 25.511345045370646],
    [15.812857543259963, 23.838221797173016],
    [14.300867357160923, 12.492638732037042],
    [39.466744493713314, 28.560193833337667],
    [14.904231064734269, 18.715023970689742],
    [7.268642631580441, 43.364491262547574],
    [38.51405372122561, 8.674461689022905],
    [29.214655667040162, 17.296978963472316],
    [4.184978711161069, 25.135982121253235],
    [9.128673564970875, 16.426561174300396],
    [17.556955605630755, 6.548494188673187],
    [18.1028591182772, 2.3238030641137275],
    [3.58621206376184, 47.44831404188169],
    [24.099622575243345, 19.678967469660762],
    [39.03397510710879, 23.131649631522972],
    [34.00818231921114, 13.603201983874163],
    [20.500273593320706, 29.95746471237647],
    [44.758867650413514, 27.443188636926504],
    [42.3354913477108, 49.06831291448991],
    [4.3025088831382, 6.416028620946024],
    [35.594933913254145, 4.281729552327413],
    [39.951358141200075, 31.08932672948325],
    [43.770490678965174, 30.060768119990723],
    [32.41263481401296, 18.499475517799485],
    [45.406144857795766, 5.714454571203525],
    [28.704343185009513, 0.3591701113047019],
    [12.166989901653096, 20.22732054217748],
    [43.44006369671841, 7.670663228971142],
    [24.55205859479325, 34.38329709893903],
    [19.48905638096563, 10.289862450699966],
    [23.869364753129, 31.43259612665422],
    [5.233802134515209, 0.7967746307855461],
    [28.671480328907656, 3.437739890824104],
    [0.24599689127488555, 43.77998915520684],
    [38.79193607337877, 38.6219005932974],
    [40.52572437868218, 30.111664020023778],
    [38.24647466395514, 31.946287497387992],
    [5.673381119635246, 46.202415294930596],
    [6.089541175612995, 13.47080182255162],
    [45.36603310734648, 6.271295304511187],
    [17.762155404014806, 6.838657275249038],
    [45.45017632148947, 20.0227655701542],
    [31.30696009512257, 13.380071887727501],
    [21.971620524973577, 4.775596570559454],
    [48.433845303436215, 25.64740552688185],
    [16.79247517389574, 43.733049248565514],
    [3.7302191014140282, 32.55369321975683],
    [27.794874129128694, 36.98500103205921],
    [44.71004137268669, 24.063685146642456],
    [19.52943147601441, 32.71525880628684],
    [42.47309247544384, 45.240327757062204],
    [19.47094539017437, 5.426017784311815],
    [6.43981963518237, 10.997497987797866],
    [38.59639961768221, 9.92423182596704],
    [24.433510027125383, 31.268211963243935],
    [24.995332954015065, 21.468503420896646],
    [31.95427429917442, 35.30828212073674],
    [5.701804924825388, 41.519772046070294],
    [17.323823188578878, 44.60291059707618],
    [18.694630889690426, 44.755836747834486],
    [15.237230869054558, 42.59755408680283],
    [11.334207101815435, 48.63220732742412],
    [42.39718004791666, 4.389110209350439],
    [5.872790429664754, 14.734000452811353],
    [18.933379424461222, 22.0565047582475],
    [11.777317706478863, 4.640551924427555],
    [36.226340975224566, 41.45696918284546],
    [32.22418305329411, 2.3384849204510507],
    [27.180953082513483, 7.415493437127601],
    [40.36235525770207, 17.124312972262118],
    [3.7037443455292585, 19.586717278980505],
    [14.889513791518432, 12.112745665219805],
    [19.90918942093781, 42.087682266354165],
    [21.905511852816982, 48.15321432866169],
    [38.86987942936344, 6.072359333716626],
    [2.544473750937165, 20.8888856681204],
    [48.671371828553376, 20.58489729794544],
    [45.348123259824916, 12.24495648682637],
    [18.646517437055305, 21.342446203425975],
    [5.322237459117051, 18.715446095102013],
    [12.915484094813074, 3.6927252161734705],
    [49.0000171282364, 10.043570813709508],
    [27.934406551168816, 1.9430804958926517],
    [16.311929125363285, 40.89770152557776],
    [22.404355862094587, 6.544361320459652],
    [4.885012386235211, 6.379856137186135],
    [1.5575379908506493, 34.15955132622241],
    [17.30851131566174, 38.04431411692017],
    [42.1596606687383, 45.66464566962805],
    [0.7174802571979666, 20.575694796713062],
    [13.81202994452056, 3.196216319976358],
    [46.54666241108046, 7.417324192779972],
    [48.15327033045268, 42.19432262313373],
    [37.47019709061107, 39.05340750682657],
    [29.08631320566962, 46.57971561596706],
    [23.01461146764787, 11.321701707321596],
    [20.27757932812045, 14.227164054187918],
    [47.45091980361487, 33.98328555222457],
    [45.57686184338961, 9.141826025595762],
    [21.409375297368005, 34.36879118696288],
    [36.04736162123352, 32.602829307094424],
    [31.358019849368933, 18.87732018274387],
    [0.8219757936780259, 42.877452981664575],
    [12.461079600938513, 40.915481540692795],
    [39.32713784018315, 41.29041089007545],
    [40.00102108030432, 19.683716050059363],
    [11.215260398951548, 42.16817067392588],
    [45.43611961464382, 15.308918127311172],
    [12.313941205349243, 35.78654182284355],
    [33.32191724128049, 8.242065989063596],
    [3.523000750603522, 24.407509750229128],
    [48.112637945130324, 48.693273169262476],
    [40.07269668048482, 29.80699107359907],
    [36.43664521981466, 43.78237210719091],
    [23.479299511507367, 48.53588012898069],
    [17.688899654003755, 10.319411269642249],
    [25.195583948130075, 42.9701844967604],
    [25.906912775591966, 29.81313186625769],
    [20.30148407634241, 25.964728040260304],
    [1.2706460791854945, 10.92691566942936],
    [18.77894110618617, 10.8413569859994],
    [46.39787446858147, 28.95830436931306],
    [16.779397038154197, 20.049749630360786],
    [2.91959154905706, 11.176248766988795],
    [33.26410829885207, 13.07139540435453],
    [40.97773655917042, 43.590302424349126],
    [48.849679265720816, 27.486087095199718],
    [21.154510114194476, 49.01865785022465],
    [40.85203016666585, 19.957510886953067],
    [46.81033076886983, 9.635744153186543],
    [5.81637797805537, 3.2832340394291504],
    [30.675773213605815, 34.10413205310003],
    [4.528589353382212, 4.542740851718735],
    [14.677681115389618, 11.259865376599027],
    [27.609460978764062, 41.30057888614114],
    [25.162102745983383, 30.44961372908943],
    [7.115498530316683, 12.669941087776039],
    [15.040712853113675, 8.04696219408948],
    [7.410593158832257, 3.586714275474506],
    [23.504536937577697, 35.92811445263253],
    [6.461867357540113, 24.25729374908588],
    [39.74639962217468, 16.470676098117192],
    [31.19146978984434, 37.84982885686393],
    [6.329254646568944, 20.305823131602885],
    [31.390560467424848, 45.37259085930275],
    [41.24168613673017, 23.389909266420204],
    [35.834293110059114, 43.06396421771106],
    [5.679000076195317, 29.153079628524498],
    [12.703568692409156, 28.431308620466726],
    [16.565628823103612, 1.8034825682334554],
    [23.568430956302866, 22.4105799206754],
    [25.662634249477396, 20.59373401883393],
    [10.696901496102596, 36.83452737360133],
    [6.48366567886981, 46.189213196059036],
    [15.86417771459322, 12.938676956374106],
    [23.5946643364745, 7.4207756293617475],
    [1.9526936687484453, 11.487310980890742],
    [20.043371329204412, 30.634471565607036],
    [22.25226416062014, 39.86863360123613],
    [1.2401610475577474, 44.95978365729265],
    [44.66656060143123, 46.494505718832656],
    [31.397446938758222, 15.341400459814414],
    [5.730070678096694, 49.64944278736543],
    [40.271091363164246, 18.24200207112013],
    [22.02254672216306, 26.6958102566042],
    [31.519366859376525, 35.92212930418924],
    [11.937727502290741, 15.703104353523024],
    [1.9733534139650344, 21.47458554636583],
    [36.27834399565729, 47.93533804821515]
   ])
tags = []
for positions in tag_positions:
    tag = Tags(positions)
    tags.append(tag)
print(len(tags))
# tags = [Tags(np.random.rand(2) * [50, 50]) for _ in range(500)]
# for tag in tags:
#     print(f"[{tag.position[0]}, {tag.position[1]}],")    
