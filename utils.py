import numpy as np

RFID_RADIUS = 3.69
def calculate_inertia_weight(w_max, w_min, iter, iter_max):
    return w_max - ((w_max - w_min) / iter_max) * iter

def calculate_covered_tags(readers, tags, rfid_radius=RFID_RADIUS):
    total_tag = len(tags)
    covered_tags = 0
    for tag in tags:
        # for reader in readers:
        #     dist = np.linalg.norm(tag.position - reader.position)
        #     if dist <= rfid_radius:
        #         tag_covered = True
        #         break
        # if tag_covered:
        #     covered_tags += 1
        if any(np.linalg.norm(tag.position - reader.position) <= rfid_radius for reader in readers):
            covered_tags +=1
    COV = (covered_tags / total_tag) * 100
    return COV

def calculate_overlap_count(readers, tags, rfid_radius):
    """
    Tính tổng số lần chồng lấp giữa các đầu đọc đối với các thẻ trong phạm vi bán kính phủ sóng.

    Tham số:
    readers: Danh sách các đầu đọc (mỗi đầu đọc có thuộc tính position)
    tags: Danh sách các thẻ (mỗi thẻ có thuộc tính position)
    rfid_radius: Bán kính phủ sóng của đầu đọc

    Trả về:
    overlap_count: Tổng số trường hợp chồng lấp giữa các đầu đọc trên các thẻ
    """
    overlap_count = 0
    for tag in tags:
        # Đếm số lượng đầu đọc phủ sóng một thẻ trong bán kính
        covering_readers = 0
        for reader in readers:
            dist = np.linalg.norm(tag.position - reader.position)
            if dist <= rfid_radius:
                covering_readers += 1  # Đếm số đầu đọc trong phạm vi phủ sóng

        # Nếu có nhiều hơn một đầu đọc phủ sóng thẻ, tính số chồng lấp
        if covering_readers > 1:
            overlap_count += covering_readers - 1  # Số lượng chồng lấp (bỏ qua đầu đọc đầu tiên)
    
    return overlap_count

def calculate_overlap_penalty(readers, rfid_radius, epsilon=1e-6):
    """
    Tính toán hình phạt nghịch đảo cho các đầu đọc chồng lấn.

    Tham số:
    readers: Danh sách các đầu đọc (đối tượng Readers).
    rfid_radius: Bán kính phủ sóng của mỗi đầu đọc.
    epsilon: Giá trị rất nhỏ để tránh chia cho 0.

    Trả về:
    inverse_penalty: Giá trị nghịch đảo của hình phạt chồng lấn cho tất cả các cặp đầu đọc.
    """
    overlap_penalty = 0
    for i in    range(len(readers)):
        for j in range(i + 1, len(readers)):
            distance = np.linalg.norm(readers[i].position - readers[j].position)
            if distance < 2 * rfid_radius:  # Kiểm tra nếu 2 đầu đọc chồng lấn
                overlap_area = (2 * rfid_radius - distance)  # Độ chồng lấn
                overlap_penalty += overlap_area  # Tăng giá trị hình phạt theo độ chồng lấn

    # Trả về nghịch đảo của giá trị hình phạt (cộng epsilon để tránh chia cho 0)
    inverse_penalty = 1 / (overlap_penalty + epsilon)
    return inverse_penalty

def calculate_interference_basic(readers, tags, rfid_radius):
    """
    Tính toán nhiễu đơn giản, chỉ kiểm tra số lượng ăng-ten phủ sóng cho mỗi thẻ.

    Tham số:
    readers: Danh sách các ăng-ten (đối tượng Readers)
    tags: Danh sách các thẻ RFID (đối tượng tags)
    rfid_radius: Bán kính phủ sóng của mỗi ăng-ten

    Trả về:
    ITF: Giá trị nhiễu tổng cộng (số thẻ bị phủ bởi hơn 1 ăng-ten)
    """
    ITF = 0  # Tổng giá trị nhiễu
    for tag in tags:  # Duyệt qua tất cả các thẻ
        antennas_covering_tag = sum(1 for reader in readers 
                                        if np.linalg.norm(tag.position - reader.position) <= rfid_radius)
        if antennas_covering_tag > 1:  # Nếu có hơn 1 ăng-ten phủ sóng thẻ
            ITF += (antennas_covering_tag - 1)  # Mỗi ăng-ten dư gây nhiễu
    return ITF

def fitness_function_basic(COV, ITF): # 23.optimizing_radio
    """
    Tính toán hàm mục tiêu đơn giản dựa trên độ phủ và nhiễu.
    
    Tham số:
    COV: Phần trăm độ phủ của mạng
    ITF: Giá trị nhiễu của mạng (số lượng ăng-ten gây nhiễu)
    OLP: Giá trị hình phạt chồng lấn giữa các đầu đọc.
    w1, w2, w3: Trọng số cho các thành phần độ phủ, nhiễu và hình phạt chồng lấn.
    Trả về:
    Giá trị hàm mục tiêu
    """
    fitness = (100 / (1 + (100 - COV) ** 2)) * 0.4 + (100 / (1 + ITF ** 2)) * 0.5
    return fitness



def fitness(tags_covered, total_tags, overlap_points, total_readers, readers_used):
    X = tags_covered / total_tags
    Y = (total_readers - readers_used) / total_readers
    Z = (total_tags - overlap_points) / total_tags
    return 0.6 * X + 0.2 * Y + 0.2 * Z

def calculate_coverage(readers, tags, rfid_radius=RFID_RADIUS):
    """
    Tính toán độ phủ sóng dựa trên vị trí của các đầu đọc và thẻ RFID.
    """
    for tag in tags:
        covered = any(np.linalg.norm(tag.position - reader.position) <= rfid_radius for reader in readers)
        if not covered:
            return False  # Có thẻ không được phủ sóng
    return True  # Tất cả thẻ đều được phủ sóng



def tentative_reader_elimination(readers, tags, coverage_function, max_recover_generations):
    """
    Hàm Tentative Reader Elimination (TRE)

    Parameters:
    - readers: danh sách đầu đọc hiện tại trong mạng, mỗi đầu đọc có thông tin về vị trí và công suất phát.
    - tags: danh sách các thẻ RFID trong khu vực, chứa thông tin về trạng thái phủ sóng của thẻ.
    - coverage_function: hàm tính toán độ phủ sóng của mạng dựa trên đầu đọc và thẻ hiện tại.
    - max_recover_generations: số thế hệ tối đa để hệ thống khôi phục độ phủ sóng nếu giảm do loại bỏ đầu đọc.

    Returns:
    - readers: danh sách đầu đọc sau khi áp dụng TRE (có thể ít hơn hoặc giữ nguyên).
    """
    # Bước 1: Kiểm tra độ phủ sóng hiện tại (bắt buộc phải có độ phủ 100%)
    full_coverage = coverage_function
    
    if not full_coverage:
        print("Không thể thực hiện TRE vì không có độ phủ sóng đầy đủ.")
        return readers
    
    # Bước 2: Chọn đầu đọc có số lượng thẻ ít nhất trong phạm vi
    readers_with_fewest_tags = sorted(readers, key=lambda reader: reader['covered_tags'])

    # Loại bỏ đầu đọc phủ sóng ít thẻ nhất
    reader_to_remove = readers_with_fewest_tags[0]
    readers.remove(reader_to_remove)
    print(f"Đã loại bỏ đầu đọc {reader_to_remove['id']}.")

    # Bước 3: Kiểm tra độ phủ sóng sau khi loại bỏ đầu đọc
    for generation in range(max_recover_generations):
        full_coverage = coverage_function(readers, tags)
        
        if full_coverage:
            print(f"Đã khôi phục độ phủ sóng đầy đủ sau {generation + 1} thế hệ.")
            return readers  # Loại bỏ đầu đọc thành công
        
        # # Giả lập quá trình tối ưu hóa qua các thế hệ (nếu có)
        # optimize_readers(readers, tags)  # Hàm này có thể là quá trình tìm kiếm giải pháp

    # Nếu không thể đạt độ phủ đầy đủ, khôi phục đầu đọc đã loại bỏ
    readers.append(reader_to_remove)
    print(f"Đã khôi phục đầu đọc {reader_to_remove['id']} vì không thể đạt độ phủ sóng sau {max_recover_generations} thế hệ.")
    
    return readers

def constrain_velocity(velocity, upper_limits, lower_limits, c1=1.5, c2=1.5):
    """
    Giới hạn vận tốc của các hạt để tránh hiện tượng bùng nổ vận tốc trong SMPSO.

    Parameters:
    - velocity: Mảng numpy chứa vận tốc hiện tại của các hạt (shape: [n_particles, n_dimensions]).
    - upper_limits: Giới hạn trên của không gian tìm kiếm cho từng chiều (array hoặc float).
    - lower_limits: Giới hạn dưới của không gian tìm kiếm cho từng chiều (array hoặc float).
    - c1, c2: Hệ số học tập nhận thức và xã hội.

    Returns:
    - Vận tốc đã được giới hạn.
    """
    
    # Tính hệ số giới hạn vận tốc (chi) dựa trên c1 và c2
    rho = c1 + c2
    if rho > 4:
        chi = 2 / (2 - rho - np.sqrt(rho**2 - 4*rho))
    else:
        chi = 1

    # Tính giới hạn vận tốc cho từng chiều
    delta = (upper_limits - lower_limits) / 2.0

    # Giới hạn vận tốc cho các hạt bằng cách áp dụng hệ số chi
    constrained_velocity = np.clip(velocity * chi, -delta, delta)
    
    return constrained_velocity