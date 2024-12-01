from classes import Tags, SSPSO, Readers
from functions import selection_mechanism, mainOptimization, BieuDoSoSanh, initialize_readers_with_kmeans, calculate_covered_tags,calculate_interference_basic,Reader_GRID, fitness_function_basic
import numpy as np
NUM_TAGS = 265
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 3
if __name__ == "__main__":
    tag_positions = np.array([
    [22.54638082, 9.8100662],
    [32.53380401, 31.04790261],
    [26.02562863, 39.58537474],
    [49.42794055, 35.8750265],
    [7.52018443, 21.66834557],
    [8.76491851, 6.95629708],
    [24.03539547, 18.44289323],
    [39.6257227, 17.42345784],
    [15.5111089, 29.75548567],
    [23.82004914, 34.37319474],
    [29.18773679, 13.71156139],
    [33.87611926, 7.94182378],
    [31.81708137, 40.03855615],
    [40.52126477, 48.60390984],
    [29.89272055, 23.16411825],
    [23.13347644, 19.01732601],
    [9.67132287, 42.82247222],
    [34.33671656, 47.93030566],
    [32.71961212, 19.97521978],
    [0.05967777, 19.81317991],
    [22.55936045, 28.59251507],
    [23.87333795, 21.54891798],
    [24.50645054, 43.23684126],
    [9.87104009, 36.07539151],
    [5.40733951, 25.26131327],
    [12.67183526, 12.92569598],
    [46.65244038, 2.4111267],
    [21.49240211, 17.60897099],
    [7.24742914, 35.73417546],
    [28.77446998, 27.32638376],
    [43.44965198, 32.2698073],
    [0.05465406, 1.35457137],
    [34.41501754, 7.07558021],
    [0.64516384, 22.22879353],
    [34.02588023, 21.97639575],
    [45.88885613, 1.7669779],
    [13.17988493, 32.66708023],
    [8.14622485, 10.38378565],
    [21.35299309, 4.55126795],
    [48.27778374, 4.32003628],
    [17.81866867, 39.80544711],
    [45.82161541, 21.98656272],
    [31.6126069, 16.36168513],
    [36.31932353, 30.4957518],
    [2.17419241, 19.94658917],
    [33.59922934, 49.95131735],
    [47.29263197, 3.39862745],
    [19.10005469, 1.61866166],
    [44.11353184, 43.20013292],
    [29.62598841, 15.82922267],
    [23.12130833, 26.74000856],
    [38.07486274, 15.0168406],
    [18.10045953, 46.72932443],
    [4.97251811, 1.01577962],
    [49.38373382, 16.73947134],
    [32.28307687, 8.89538283],
    [6.77465576, 10.94064505],
    [29.09711492, 26.51739215],
    [24.18550574, 33.40861524],
    [49.55909234, 16.44769119],
    [2.30346191, 4.35762546],
    [34.73780168, 23.45849419],
    [15.00297428, 11.99467158],
    [26.56620256, 26.01459549],
    [6.53591805, 48.34232701],
    [32.31473334, 22.17436792],
    [20.97695407, 41.56231776],
    [14.48701345, 9.67858916],
    [31.37226556, 20.85464402],
    [8.22802338, 0.28302666],
    [36.15141887, 7.83497714],
    [9.82934485, 7.01917493],
    [0.21919844, 18.32953538],
    [40.08902647, 12.669206],
    [45.92295053, 47.3022857],
    [9.96798681, 4.67649799],
    [6.38494343, 2.91669585],
    [0.58798341, 0.7585496],
    [8.56072398, 24.4032537],
    [30.98504231, 40.34120711],
    [12.56789386, 47.66586661],
    [39.1418654, 4.3591591],
    [38.78117112, 31.11192404],
    [23.98702679, 26.63552587],
    [24.58213499, 14.50720797],
    [18.03673047, 30.43340384],
    [5.14050596, 38.10533223],
    [27.80691622, 21.55594741],
    [2.9146046, 19.94223004],
    [0.56500577, 33.53213957],
    [0.13889932, 28.09299175],
    [27.40818491, 45.08473498],
    [49.45271214, 13.12499986],
    [23.33213156, 23.12022271],
    [37.56406462, 34.73387679],
    [48.39365703, 17.9829188],
    [16.30885001, 35.61447536],
    [12.89777837, 21.50901639],
    [49.86828412, 0.59694838],
    [20.74099533, 48.97156066]
   ])
    # tags = []
    # for positions in tag_positions:
    #     tag = Tags(positions)
    #     tags.append(tag)

    #readers = [Readers(np.random.rand(2) * [50, 30]) for _ in range(NUM_RFID_READERS)]
    tags = [Tags(np.random.rand(2) * [50, 50]) for _ in range(NUM_TAGS)]

 
    #readers, tracking = mainOptimization(tags, readers, sspso, 3.2, readers)
    


    
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

        reader_3_2 = sspso_3_2.optimize(tags)
        reader_1_6 = sspso_1_6.optimize(tags)
        reader_0_8 = sspso_0_8.optimize(tags)

        reader_3_2 = Reader_GRID(reader_3_2, 3.2)
        reader_1_6 = Reader_GRID(reader_1_6, 1.6)
        reader_0_8 = Reader_GRID(reader_0_8, 0.8)

        IFT_3_2 = calculate_interference_basic(reader_3_2, tags)
        IFT_1_6 = calculate_interference_basic(reader_1_6, tags)
        IFT_0_8 = calculate_interference_basic(reader_0_8, tags)
        
        COV_3_2 = calculate_covered_tags(reader_3_2, tags)
        COV_1_6 = calculate_covered_tags(reader_1_6, tags)
        COV_0_8 = calculate_covered_tags(reader_0_8, tags)

        FIT_3_2 = fitness_function_basic(COV_3_2, IFT_3_2, tags, 0.8, 0.2)
        FIT_1_6 = fitness_function_basic(COV_1_6, IFT_1_6, tags, 0.8, 0.2)
        FIT_0_8 = fitness_function_basic(COV_0_8, IFT_0_8, tags, 0.8, 0.2)
        
        tracking_3_2.append(np.array([FIT_3_2, i]))
        tracking_1_6.append(np.array([FIT_1_6, i]))
        tracking_0_8.append(np.array([FIT_0_8, i]))

    BieuDoSoSanh(tracking_3_2, tracking_1_6, tracking_0_8)
    