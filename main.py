from classes import Tags, Readers, SSPSO, GRID_X, GRID_Y
from functions import BieuDotags, BieuDoReader, RFID_RADIUS, selection_mechanism, mainOptimization

NUM_INDIVIDUALS = 100
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 3
if __name__ == "__main__":
    tags = [Tags(DIM) for _ in range(NUM_INDIVIDUALS)]
    readers = selection_mechanism(tags, NUM_RFID_READERS)
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, readers)
    mainOptimization(tags, readers, sspso)


