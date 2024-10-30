from classes import Tags, Readers, SSPSO
from functions import BieuDotags, BieuDoReader, RFID_RADIUS
NUM_RFID_READERS = 25
NUM_INDIVIDUALS = 500
NUM_ITERATION = 50
DIM = 2
readers =  [Readers(DIM) for _ in range(NUM_RFID_READERS)]
tags = [Tags(DIM) for _ in range(NUM_INDIVIDUALS)]
sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION)
BieuDoReader(readers, tags)
sspso.optimize(tags, RFID_RADIUS)
BieuDoReader(readers, tags)
BieuDotags(readers, tags)
