import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from classes import Readers
from classes import  GRID_X, GRID_Y

RFID_RADIUS = 3.69
UPDATE_INTERVAL = 500
NUM_RFID_READERS = 35
DIM = 2
def initialize_readers_with_kmeans(TAGS, n_readers):
    # Lấy vị trí của các tag
    tag_positions = np.array([tag.position for tag in TAGS])
    
    # Sử dụng thuật toán k-means để tìm các cluster center
    kmeans = KMeans(n_clusters=n_readers, random_state=0).fit(tag_positions)
    
    # Đặt các reader tại các cluster center
    reader_positions = kmeans.cluster_centers_
    
    # Khởi tạo các reader với vị trí mới
    READERS = [Readers(position=pos) for pos in reader_positions]
    
    return READERS
def generate_hexagon_centers_with_boundary(width, height, cell_size=3.2, center_distance_x=6.4):
    centers = []
    dy = cell_size * math.sqrt(3)  # Khoảng cách dọc giữa các tâm (khoảng 5.54 mét)

    # Lặp qua các hàng và cột của lưới
    y = 0
    row = 0
    hexagon_count = 0  # Biến đếm số lượng lục giác
    while y <= height:
        x_offset = (row % 2) * (center_distance_x / 2)  # Dịch ngang cho hàng chẵn lẻ
        x = x_offset
        while x <= width:
            # Chỉ thêm các điểm trên biên của lưới
            if y == 0 or y + dy > height or x == 0 or x + center_distance_x > width:
                centers.append((round(x, 2), round(y, 2)))
                hexagon_count += 1
            x += center_distance_x
        y += dy
        row += 1
    return centers, hexagon_count
def generate_hexagon_centers(width, height, cell_size=3.2, center_distance_x=6.4):
    centers = []
    dy = cell_size * math.sqrt(3)  # Khoảng cách dọc giữa các tâm (khoảng 5.54 mét)

    # Lặp qua các hàng và cột của lưới
    y = 0
    row = 0
    hexagon_count = 0  # Biến đếm số lượng lục giác
    while y <= height:
        x_offset = (row % 2) * (center_distance_x / 2)  # Dịch ngang cho hàng chẵn lẻ
        x = x_offset
        while x <= width:
            centers.append((round(x, 2), round(y, 2)))
            hexagon_count += 1
            x += center_distance_x
        y += dy
        row += 1

    return centers, hexagon_count

def BieuDotags(READERS, TAGS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')
    
    tag_positions = np.array([tag.position for tag in TAGS])
    reader_positions = np.array([reader.position for reader in READERS])
    
    scatter_tag = ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color='blue', label='TAGS', s=10, marker='x')
    ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='Readers', marker='^')
    
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)
    
    count_text = ax.text(0.02, 1.05, 'Tags in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def update_tag(frame):
        for tag in TAGS:
            tag.update_position()
            tag.covered = any(np.linalg.norm(tag.position - reader.position)<= RFID_RADIUS for reader in READERS)
        
        tag_positions = np.array([tag.position for tag in TAGS])
        tag_positions = np.atleast_2d(tag_positions)

        colors = ['green' if tag.covered else 'blue' for tag in TAGS]

        scatter_tag.set_offsets(tag_positions)
        scatter_tag.set_color(colors)

        count_in_range = sum(tag.covered for tag in TAGS)
        count_text.set_text(f'Tags in range: {count_in_range}')
        print(f"{count_in_range} tags in range")

        return scatter_tag

    ani = animation.FuncAnimation(fig, update_tag, frames=999999, interval=UPDATE_INTERVAL, blit=False, repeat=False)
    plt.show()

def BieuDoReader(readers, tags):
    """
    Vẽ biểu đồ vị trí các đầu đọc và các thẻ.

    Parameters:
    - readers: Danh sách các đối tượng reader
    - tags: Danh sách các đối tượng tag
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Lấy vị trí của các tag
    tag_positions = np.array([tag.position for tag in tags])
    ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color='blue', label='Tags', s=10, marker='x')

    # Lấy vị trí của các reader có active = True
    active_reader_positions = np.array([reader.position for reader in readers if reader.active])
    ax.scatter(active_reader_positions[:, 0], active_reader_positions[:, 1], color='red', label='Active Readers', marker='^')

    # Vẽ các vòng tròn phạm vi phủ sóng của các reader có active = True
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in active_reader_positions]
    for circle in circles:
        ax.add_artist(circle)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='lower right')
    plt.show()