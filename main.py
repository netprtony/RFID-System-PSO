from classes import Tags, SSPSO, Readers
from functions import tracking_time, tracking_COV, tracking_Fitness, tracking_IFT, selection_mechanism, mainOptimization, initialize_readers_with_kmeans, tags, BieuDoSoSanh
import numpy as np
import time
NUM_TAGS = 265
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 3
import random
if __name__ == "__main__":
 
    # readers = [Readers(np.random.rand(2) * [50, 30]) for _ in range(NUM_RFID_READERS)]
    # tags = [Tags(np.random.rand(2) * [50, 50]) for _ in range(NUM_TAGS)]
    #tags = random.sample(random_tag, 99)
    readers = selection_mechanism(tags, NUM_RFID_READERS,1)
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, readers)
    sspso.readers = mainOptimization(tags, sspso, GRID_SIZE=3.2)
    

    
    # tracking_3_2 = []
    # tracking_1_6 = []
    # tracking_0_8 = []
    # for i in range(5, 51, 5):
    #     print(f"Running with {i} readers")
    #     reader_3_2 = initialize_readers_with_kmeans(tags, i)
    #     reader_0_8 = initialize_readers_with_kmeans(tags, i)
    #     reader_1_6 = initialize_readers_with_kmeans(tags, i)

    #     sspso_3_2 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_3_2)
    #     sspso_1_6 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_1_6)
    #     sspso_0_8 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_0_8)

    #     value_3_2 = tracking_time(sspso_3_2, tags, 3.2)
    #     value_1_6 = tracking_time(sspso_1_6, tags, 1.6)
    #     value_0_8 = tracking_time(sspso_0_8, tags, 0.8)

    #     tracking_3_2.append(np.array([i, value_3_2]))
    #     tracking_1_6.append(np.array([i, value_1_6]))
    #     tracking_0_8.append(np.array([i, value_0_8]))
    # BieuDoSoSanh(tracking_3_2, tracking_1_6, tracking_0_8, "Số lượng đầu đọc", "Thời gian thực hiện (giây)")