import numpy as np
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from classes import Readers, Tags
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
    """Tạo lưới các điểm trong không gian hoạt động."""
    x_coords = np.arange(0, grid_x, grid_size)
    y_coords = np.arange(0, grid_y, grid_size)
    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    return grid_points

def snap_to_grid(position, grid_points):
    """Đưa vị trí về điểm trên lưới gần nhất."""
    distances = np.linalg.norm(grid_points - position, axis=1)
    nearest_point = grid_points[np.argmin(distances)]
    return nearest_point
def Reader_GRID(readers, GRID_SIZE):
    grid_points = create_grid(GRID_SIZE, GRID_X, GRID_Y)
   #Điều chỉnh vị trí các đầu đọc về mắt lưới gần nhất
    for reader in readers:
        reader.position = snap_to_grid(reader.position, grid_points)
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

    return readers  # Trả về danh sách đầu đọc đã được chọn


def BieuDoSoSanh(tracking3_2, tracking1_6, tracking0_8):
    # Tách dữ liệu từ tracking
    COV3_2 = [item[0] for item in tracking3_2]
    readerNum3_2 = [item[1] for item in tracking3_2]
    COV1_6 = [item[0] for item in tracking1_6]
    readerNum1_6 = [item[1] for item in tracking1_6]
    COV0_8 = [item[0] for item in tracking0_8]
    readerNum0_8 = [item[1] for item in tracking0_8]
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(readerNum3_2, COV3_2, marker='^', label='Grid 3.2', color='gray', linewidth=2)
    plt.plot(readerNum1_6, COV1_6, marker='^', label='Grid 1.6', color='red', linewidth=2)
    plt.plot(readerNum0_8, COV0_8, marker='^', label='Grid 0.8', color='blue', linewidth=2)
    # Labels and legend
    plt.xlabel('Số lượng đầu đọc', fontsize=12)
    plt.ylabel('Giá trị fitness', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    # Show the plot
    plt.show()


def adjust_readers_location_by_virtual_force(readers, tags, max_no_change_iterations=50):
    no_change_iterations = 0
    best_fitness = -float('inf')
    best_positions = [reader.position.copy() for reader in readers]

    while no_change_iterations < max_no_change_iterations:
        for reader in readers:
            # Lực tổng hợp được khởi tạo
            total_exclusion_force = np.array([0.0, 0.0])
            total_attraction_force = np.array([0.0, 0.0])

            # 1. Lực đẩy (Exclusion Operator)
            for other_reader in readers:
                if other_reader != reader:
                    distance = np.linalg.norm(reader.position - other_reader.position)
                    if distance < 2 * RFID_RADIUS:  # Kiểm tra xem hai đầu đọc có chồng lấn không
                        # Tính toán lực đẩy
                        force_magnitude = EXCLUSION_FORCE * (2 * RFID_RADIUS - distance)
                        direction = (reader.position - other_reader.position) / distance
                        total_exclusion_force += force_magnitude * direction

            # 2. Lực hút (Attraction Operator)
            for tag in tags:
                if not tag.covered:  # Nếu thẻ chưa được bao phủ
                    distance = np.linalg.norm(reader.position - tag.position)
                    if distance <= RFID_RADIUS:  # Kiểm tra xem thẻ có nằm trong phạm vi hấp dẫn không
                        # Tính toán lực hút
                        force_magnitude = ATTRACTION_FORCE * (RFID_RADIUS - distance)
                        direction = (tag.position - reader.position) / distance
                        total_attraction_force += force_magnitude * direction

            # 3. Cập nhật vị trí đầu đọc dựa trên lực tổng hợp
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
    fig.text(0.3, 0.92, f"Có {coverage_tag:.0f} thẻ bao phủ trong {len(tags)}", fontsize=12, color="black", ha='left', va='top')
    fig.text(0.5, 0.92, f"Độ nhiễu: {interference:.2f}%", fontsize=12, color="orange", ha='left', va='top')
    fig.text(0.6, 0.92, f"Số lượng đầu đọc: {active_reader_count}", fontsize=12, color="blue", ha='left', va='top')

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Đưa chú thích biểu đồ ra ngoài phía trên bên phải
    # Đặt tiêu đề cho biểu đồ
    fig.suptitle(title, fontsize=14, ha='left', va='top', fontweight='bold', x=0.01, y=0.99)
    plt.show()

def BieuDoSoSanh(tracking3_2, tracking1_6, tracking0_8):
    # Tách dữ liệu từ tracking
    COV3_2 = [item[0] for item in tracking3_2]
    readerNum3_2 = [item[1] for item in tracking3_2]
    COV1_6 = [item[0] for item in tracking1_6]
    readerNum1_6 = [item[1] for item in tracking1_6]
    COV0_8 = [item[0] for item in tracking0_8]
    readerNum0_8 = [item[1] for item in tracking0_8]
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(readerNum3_2, COV3_2, marker='^', label='Grid 3.2', color='gray', linewidth=2)
    plt.plot(readerNum1_6, COV1_6, marker='^', label='Grid 1.6', color='red', linewidth=2)
    plt.plot(readerNum0_8, COV0_8, marker='^', label='Grid 0.8', color='blue', linewidth=2)
    # Labels and legend
    plt.xlabel('Số lượng đầu đọc', fontsize=12)
    plt.ylabel('Thời gian thực hiện (s)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    # Show the plot
    plt.show()

def mainOptimization(tags, sspso, GRID_SIZE):
    sspso.readers = sspso.optimize(tags, RFID_RADIUS)
    BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi tối ưu hội tụ", GRID_SIZE)
    sspso.readers = adjust_readers_location_by_virtual_force(sspso.readers, tags)
    BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi tối ưu hóa bằng lực ảo", GRID_SIZE)
    grid_points = create_grid(GRID_SIZE, GRID_X, GRID_Y)
    for reader in sspso.readers:
            reader.position = snap_to_grid(reader.position, grid_points)
    BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi đưa vị trí về mắt lưới", GRID_SIZE)       
    sspso.readers = Redundant_Reader_Elimination(sspso.readers, tags)
    BieuDoReader(sspso.readers, tags, "Biểu đồ sau khi loại bỏ đầu đọc dư thừa", GRID_SIZE)
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


tag_positions = np.array([
    [1.2725506818141274, 47.26795885127647],
    [42.3567994512635, 14.821376499938177],
    [15.720621922161476, 35.092950858457904],
    [26.003451637155667, 40.08245750098026],
    [38.49822987881422, 28.39057536206632],
    [8.281990440784398, 16.379043109100248],
    [20.67251125249433, 12.141611676345686],
    [30.120840895618926, 25.387507210099443],
    [28.041878053355703, 14.182603318777664],
    [20.89001313198078, 36.994244754966246],
    [38.462166414688326, 41.55153664094502],
    [4.215587713446012, 11.330356149189342],
    [33.9378681339111, 27.852541044104147],
    [2.4526149560318733, 28.970214613369304],
    [30.113253170542613, 26.09647317604646],
    [12.246046138590255, 46.161443609792755],
    [6.07973267343519, 29.207003718438234],
    [2.9247376072340403, 37.39305287051953],
    [8.748342153488547, 39.35447441899711],
    [26.868163520223202, 33.42638912911994],
    [48.64149274322322, 37.849290221198046],
    [23.99191111650661, 28.61454777824271],
    [15.745465862338376, 0.3608554006373721],
    [35.693540034604425, 47.84222531882884],
    [2.8328388972199803, 29.91751997246664],
    [15.925219628932009, 46.22615488367999],
    [7.483622928619171, 27.832088517341464],
    [14.872392058380996, 12.792027651710875],
    [49.21897385762685, 36.94904772744167],
    [25.188971309036358, 25.060538566329306],
    [25.15038154108162, 32.07250189464769],
    [1.3600897069835305, 13.665505861768445],
    [36.05500297416325, 36.70578411595158],
    [3.0004723137197487, 2.9816419121373228],
    [48.97881402660961, 27.688474229374243],
    [14.984622436534046, 18.287931641970346],
    [47.60874177609725, 26.85586093583476],
    [18.837646068930518, 32.33722138504981],
    [14.541158497046764, 28.601789624588612],
    [1.9127913660255125, 18.794907592710143],
    [36.78858530820369, 41.73375747987711],
    [0.25093231252210635, 46.66602650489352],
    [27.48971772571868, 23.549198386547133],
    [37.498147613757695, 41.62483807569016],
    [36.479323313040915, 9.093270595226915],
    [35.05978882707006, 18.244063485142103],
    [46.60792963787234, 41.96759095015336],
    [26.075994064654683, 34.19787557096207],
    [48.345208152978195, 42.69247803849939],
    [21.321068874347272, 18.31717728991756],
    [5.283028899171244, 21.255548350259172],
    [13.209261861711857, 23.69887016164326],
    [14.616749120140915, 15.0976641263738],
    [39.03459038104109, 3.1567795184061884],
    [13.672741855606663, 16.86716286359343],
    [4.6675466309201274, 15.549857412739005],
    [1.827399103961258, 13.22886982480546],
    [47.07340244683895, 41.69628102878388],
    [48.053866663307566, 7.833419809020886],
    [33.29150514277065, 40.687069573203416],
    [20.85146481404768, 9.451371741432407],
    [32.0611986354818, 43.4506236697832],
    [41.33953675567179, 39.74789257751701],
    [4.12413085872973, 11.620767166890156],
    [30.890100798099517, 47.29812966449201],
    [44.535497215678646, 42.518264260155846],
    [15.11766898391969, 23.132954339181744],
    [31.836626660150685, 0.34649047764019336],
    [13.85568074088097, 21.789231171244978],
    [13.281560615217591, 8.892371875687838],
    [1.0787564989456822, 4.340141616729798],
    [23.23273736335192, 10.377537746782817],
    [41.29839712427382, 9.111262181849817],
    [8.687021918127469, 45.26079116623028],
    [39.53859675718321, 0.032763092109200365],
    [40.95503676539098, 32.09889091777895],
    [38.60601262980406, 23.84824067224779],
    [26.034673514216088, 41.049371315783446],
    [28.492185899878482, 8.586655901621615],
    [20.466144140889302, 8.705984034658453],
    [9.911244168111821, 15.503981678545253],
    [16.107049968286635, 10.839567654660431],
    [16.463917281937057, 30.3393209457738],
    [37.591856264495036, 41.6935943614382],
    [30.378959557254614, 28.51828222573209],
    [33.72792235828858, 11.200657100463024],
    [20.45808681300493, 9.613081369056953],
    [32.24970153167745, 31.35341153170186],
    [32.63449603588504, 0.1987364644826295],
    [11.29745326982584, 35.43290537960437],
    [12.127443496738211, 9.777992901116328],
    [46.4302575337956, 22.997447246756032],
    [8.482745808119429, 21.61092338638127],
    [9.232814854212606, 19.706250266986974],
    [8.795850617732226, 17.071354094848264],
    [7.811501937051729, 27.76376931461748],
    [22.391198033339833, 17.248918208240628],
    [30.712703001219843, 21.737781403996408],
    [20.05845520463559, 15.950252751383493],
    [39.44882538535734, 8.624307252916113],
    [7.535753451743266, 12.19749066569888],
    [16.785081212771196, 30.75759407860409],
    [21.5900822389126, 47.74811465002282],
    [32.82658782598258, 44.37533711036438],
    [17.173296396738063, 29.90573613237516],
    [1.5057527618286437, 2.908103603973089],
    [14.200947301255319, 12.249861484372582],
    [11.986125079279148, 32.35355971390435],
    [16.285788447302096, 30.843000695679102],
    [17.758894233731848, 49.31738991802928],
    [0.027647783037620943, 2.0912533611541884],
    [25.692246954928322, 20.241658507077492],
    [22.27494591192447, 4.201442931039851],
    [13.125322958649221, 9.793001916564814],
    [30.236605114231672, 27.177144698917598],
    [8.659054434863506, 24.37252142563221],
    [21.78216949080382, 22.15539385400153],
    [18.75490354730436, 42.840507882386845],
    [42.38261625161731, 30.1998434139566],
    [42.0904925920656, 47.742112301958805],
    [31.05340965616847, 19.277193508581814],
    [16.789461147910885, 13.389026717452762],
    [44.200185641879536, 9.700302054582439],
    [39.281387402093394, 30.940861230326945],
    [42.88481264272617, 11.351008977231803],
    [11.02755229486076, 25.44025813814474],
    [25.91772922720536, 14.243656895576246],
    [38.75966266233846, 6.488520495376021],
    [4.174138919866543, 19.58367809186295],
    [45.267724628087834, 14.421273683319624],
    [42.26395617729493, 15.085490351332398],
    [36.79298173253892, 31.90653586867962],
    [20.726114303782943, 6.388743248480772],
    [23.2082810133717, 34.05804973087336],
    [4.577086543808518, 1.3306866051248978],
    [34.461813199313625, 11.524253117762823],
    [48.66621290032112, 23.60527450069782],
    [16.006832821721122, 24.242741141735785],
    [36.28275298406044, 20.29606156918288],
    [18.181339427075788, 40.958814251087674],
    [33.599942808580366, 9.665980092814818],
    [33.175873871915414, 38.26042151529162],
    [3.680216674056042, 11.362242108169362],
    [26.078939596057765, 49.12592696801018],
    [23.7039495052182, 23.62253332735147],
    [43.68195508803382, 45.64450625980745],
    [32.60146200201226, 24.25132186606999],
    [43.362754885570766, 45.721860630118144],
    [31.678435741506526, 7.944805497600354],
    [4.674658572175083, 41.566755880686415],
    [36.22817755821917, 18.85061583533053],
    [15.168925600339279, 44.662392167728115],
    [23.8384944577487, 24.542097118905275],
    [45.39790606313493, 40.3567797222389],
    [12.906355721281942, 1.5840747971342495],
    [48.724905106309926, 19.400377671146984],
    [29.620595216750335, 16.06493931368017],
    [22.156775856849187, 23.480014070466193],
    [38.656114364562846, 15.840683301278258],
    [14.853186926458816, 47.73928624604829],
    [46.16451457347662, 22.836128727612927],
    [21.22383770802106, 10.609940827442083],
    [30.61061218824076, 27.010030814772897],
    [0.586976651336657, 6.274025260476118],
    [29.89844171825929, 18.45739858691867],
    [17.716906634340866, 29.386846399292626],
    [21.44522652317783, 49.485445234046495],
    [16.01721934571998, 27.37063793442574],
    [8.524859987105348, 10.16940885183737],
    [28.332269446646418, 16.56291585506335],
    [21.85815095974303, 8.299879303564817],
    [43.30033652293885, 8.071658183609326],
    [2.1814011397611077, 35.65905786547571],
    [45.56983227995023, 16.48639556031195],
    [22.780675000987443, 6.293355634941405],
    [27.845153843115305, 19.705283789758337],
    [10.165745716860592, 6.46368363183909],
    [30.791974311482445, 31.37627557611065],
    [20.42461647794073, 14.109255023108213],
    [16.858898477967138, 21.41961114598578],
    [8.323878407634687, 49.13086932731377],
    [1.776535130249085, 38.615329543764524],
    [27.76319184439967, 47.123432188137635],
    [27.975360719192143, 46.04316489589006],
    [23.817669629862, 40.37214125614318],
    [5.764063903886485, 17.929286955099542],
    [15.531159398229665, 17.579607955132833],
    [31.087001042553158, 8.665509138477374],
    [4.26442759570948, 35.02326616392386],
    [20.568942313515485, 30.039038833424385],
    [25.483249776720907, 41.504456298158765],
    [35.511017732910496, 7.292742425391458],
    [28.21409983955728, 22.32408919139245],
    [34.736298524076346, 38.180047810142916],
    [34.19955842971912, 22.49316093109372],
    [4.70679887805015, 39.7103033536392],
    [24.845169169191223, 1.47603785572602],
    [33.7890841659491, 46.901766962398725],
    [47.60588133183386, 33.709370109009875],
    [43.597143158171505, 10.598976944598364],
    [34.76197272797702, 28.23461278968948],
    [17.77826044718052, 6.009079497333259],
    [9.898781832001863, 22.644505965704564],
    [22.647355372884505, 16.227459560980627],
    [25.650734336344115, 20.35183425604376],
    [24.165673427827755, 14.214315092864716],
    [45.561734128353685, 44.298675401006946],
    [18.05214977362174, 2.363590029811319],
    [8.786450049181388, 25.59014958327397],
    [6.413731505868214, 43.054935546643044],
    [23.3292622020551, 10.901596839267736],
    [49.29067833339345, 27.010588358123595],
    [49.3944318185636, 9.434381850646284],
    [40.198050327859555, 42.78042922607824],
    [7.807613637892608, 38.58489045859372],
    [12.684939958351988, 10.790127996203418],
    [8.447591022234462, 27.598299959229006],
    [48.564150390269575, 12.217270587700247],
    [0.19596512475356098, 2.9214626439941824],
    [31.751826903902085, 0.5718820346516273],
    [14.667082525022584, 21.60656457479992],
    [27.806964502285137, 5.373924324821344],
    [10.22713360048102, 0.026336987932623623],
    [29.565364792940574, 31.353217948708583],
    [41.6051712419648, 44.85209287548226],
    [31.200127658189636, 14.87879820360784],
    [17.203284382775262, 29.018711141618642],
    [33.699424232333776, 34.21690638639906],
    [14.034491278909995, 7.632507704207125],
    [32.058954770967496, 9.432111104547808],
    [9.845971649019807, 29.974205461223402],
    [16.72947055522998, 49.562756601838124],
    [38.53200034492749, 16.10211727240733],
    [45.08646948596249, 9.543056247034738],
    [22.914856202569574, 35.02447685081467],
    [7.970692986769906, 33.49758162948594],
    [11.942634873223568, 32.30260230872097],
    [35.937228024658964, 41.609607511244164],
    [47.81891304235721, 17.458322203973204],
    [6.0255532981368205, 29.71532816459642],
    [20.72411567308266, 30.02598832966993],
    [11.240983266692266, 45.964508267330096],
    [33.3880175217492, 8.232409475757734],
    [25.868499593916027, 26.337339802602038],
    [46.33878734524555, 8.951349818837734],
    [47.6844460250178, 29.74842584371027],
    [20.90707957224029, 9.327623701750165],
    [3.1107696361384827, 9.805412759941767],
    [37.44311105177272, 8.888580477626796],
    [31.221349315275493, 19.38122664115055],
    [34.40960987707427, 13.445476445307158],
    [19.868025337884838, 28.647109270347045],
    [14.47166665297842, 33.43810733258471],
    [41.629717894490646, 36.4305472896997],
    [36.2530194470486, 3.199326329739166],
    [30.975923392973886, 7.560375976872896],
    [8.440040227239226, 46.8078358874218],
    [32.48862183447731, 0.3339967836259894],
    [23.96145642394629, 2.908036796912822],
    [0.9501581771276468, 34.289792238851796],
    [17.85720050635917, 10.756448784738692],
    [7.35641622734105, 14.553766186808614],
    [2.7646850851669957, 7.797296278606408],
    [32.73456186852333, 22.714567646596564],
    [9.175618634518523, 31.645734925109288]
   ])
tags = []
for positions in tag_positions:
    tag = Tags(positions)
    tags.append(tag)



