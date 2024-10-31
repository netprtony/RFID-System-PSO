import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from classes import  GRID_X, GRID_Y

RFID_RADIUS = 3.69
GRID_X, GRID_Y = 30, 25 
UPDATE_INTERVAL = 500
NUM_RFID_READERS = 35
DIM = 2


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

def BieuDoReader(READERS, TAGS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Trích xuất vị trí sinh viên và đầu đọc RFID
    tag_positions = np.array([tag.position for tag in TAGS])
    reader_positions = np.array([reader.position for reader in READERS])
    
    ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color='blue', label='Tags', s=10 , marker='x')
    scatter_rfid = ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='RFID Readers', marker='^')

    # Tạo các hình tròn biểu diễn vùng phủ sóng của các đầu đọc RFID
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)

    # Text hiển thị số sinh viên trong vùng bán kính
    count_text = ax.text(0.02, 1.05, 'Tags in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    count_in_range = 0
    for tag in tag_positions:
        if any(np.linalg.norm(tag - reader.position) <= RFID_RADIUS for reader in READERS):
            count_in_range += 1  

    count_text.set_text(f'Tags in range: {count_in_range}')    
    scatter_rfid, *circles
    plt.show()
    