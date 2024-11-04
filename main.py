from classes import Tags, Readers, SSPSO, GRID_X, GRID_Y
from functions import BieuDotags, BieuDoReader, RFID_RADIUS, initialize_readers_with_kmeans

NUM_INDIVIDUALS = 500
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 30
tags = [Tags(DIM) for _ in range(NUM_INDIVIDUALS)]
readers = initialize_readers_with_kmeans(tags, NUM_RFID_READERS)
sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, readers)

BieuDoReader(readers, tags)
readers = sspso.optimize(tags, RFID_RADIUS)
BieuDoReader(readers, tags)
#BieuDotags(readers, tags)
