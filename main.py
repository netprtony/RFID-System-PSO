from classes import Tags, SSPSO, Readers
from functions import selection_mechanism, mainOptimization, BieuDoSoSanh, initialize_readers_with_kmeans, calculate_covered_tags,calculate_interference_basic,Reader_GRID, fitness_function_basic, tags
import numpy as np
import time
NUM_TAGS = 265
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 3
if __name__ == "__main__":
 
    # readers = [Readers(np.random.rand(2) * [50, 30]) for _ in range(NUM_RFID_READERS)]
    # tags = [Tags(np.random.rand(2) * [50, 50]) for _ in range(NUM_TAGS)]
    
    # readers = selection_mechanism(tags, NUM_RFID_READERS, 0.95)
    # sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, readers)
    # sspso.readers = mainOptimization(tags, sspso, GRID_SIZE=3.2)
    

    
    tracking_3_2 = []
    tracking_1_6 = []
    tracking_0_8 = []
    for i in range(5, 51):
        reader_3_2 = initialize_readers_with_kmeans(tags, i)
        reader_0_8 = initialize_readers_with_kmeans(tags, i)
        reader_1_6 = initialize_readers_with_kmeans(tags, i)

        sspso_3_2 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_3_2)
        sspso_1_6 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_1_6)
        sspso_0_8 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_0_8)

        start_time_3_2 = time.time()
        reader_3_2 = sspso_3_2.optimize(tags)
        reader_3_2 = Reader_GRID(reader_3_2, 3.2)
        end_time_3_2 = time.time()

        start_time_1_6 = time.time()
        reader_1_6 = sspso_1_6.optimize(tags)
        reader_1_6 = Reader_GRID(reader_1_6, 1.6)
        end_time_1_6 = time.time()

        start_time_0_8 = time.time()
        reader_0_8 = sspso_0_8.optimize(tags)
        reader_0_8 = Reader_GRID(reader_0_8, 0.8)
        end_time_0_8 = time.time()
        
        
        
        

        
        
        tracking_3_2.append(np.array([int(end_time_3_2 - start_time_3_2), i]))
        tracking_1_6.append(np.array([int(end_time_1_6 - start_time_1_6), i]))
        tracking_0_8.append(np.array([int(end_time_0_8 - start_time_0_8), i]))
    BieuDoSoSanh(tracking_3_2, tracking_1_6, tracking_0_8)