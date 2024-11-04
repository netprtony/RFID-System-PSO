import numpy as np

RFID_RADIUS = 3.69
def calculate_inertia_weight(w_max, w_min, iter, iter_max):
    return w_max - ((w_max - w_min) / iter_max) * iter

def calculate_covered_tags(readers, tags, rfid_radius=RFID_RADIUS):
    total_tag = len(tags)
    covered_tags = 0
    for tag in tags:
        if any(np.linalg.norm(tag.position - reader.position) <= rfid_radius for reader in readers):
            covered_tags +=1
    COV = (covered_tags / total_tag) * 100
    return COV

def calculate_uncovered_tags(tags, readers, rfid_radius):
    """
    Tính toán số lượng các tag không được bao phủ bởi bất kỳ reader nào.

    Parameters:
    - tags: Danh sách các đối tượng tag
    - readers: Danh sách các đối tượng reader
    - rfid_radius: Bán kính phủ sóng của reader

    Returns:
    - num_uncovered_tags: Số lượng các tag không được bao phủ
    """
    num_uncovered_tags = 0

    for tag in tags:
        covered = any(np.linalg.norm(tag.position - reader.position) <= rfid_radius for reader in readers)
        if not covered:
            num_uncovered_tags += 1

    return num_uncovered_tags


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

def constrain_velocity(velocity, upper_limit, lower_limit):
    """
    Giới hạn tốc độ của hạt trong phạm vi đã cho.

    Parameters:
    - velocity: Vận tốc hiện tại của hạt (danh sách hoặc mảng chứa các giá trị vận tốc ở từng chiều)
    - upper_limit: Giới hạn trên của không gian tìm kiếm (danh sách hoặc mảng cùng độ dài với velocity hoặc số nguyên)
    - lower_limit: Giới hạn dưới của không gian tìm kiếm (danh sách hoặc mảng cùng độ dài với velocity hoặc số nguyên)

    Returns:
    - velocity: Vận tốc đã được giới hạn
    """
    # Nếu upper_limit và lower_limit là số nguyên, chuyển đổi chúng thành danh sách
    if isinstance(upper_limit, (int, float)):
        upper_limit = [upper_limit] * len(velocity)
    if isinstance(lower_limit, (int, float)):
        lower_limit = [lower_limit] * len(velocity)

    # Tính giá trị giới hạn tốc độ delta
    delta = [(u - l) / 2 for u, l in zip(upper_limit, lower_limit)]

    # Giới hạn vận tốc
    for i in range(len(velocity)):
        if velocity[i] > delta[i]:
            velocity[i] = delta[i]
        elif velocity[i] < -delta[i]:
            velocity[i] = -delta[i]

    return velocity