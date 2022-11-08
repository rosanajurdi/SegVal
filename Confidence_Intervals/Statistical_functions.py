from re import A
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
from datetime import date

def statistical_analysis(df):
    "function that getd as input a pandas dataframe and produced some statistical values:"
    mean = np.round(df.mean().values[0], 4) # get the mean of the sample of size k
    std = np.round(df.std().values[0], 4) # get the standard deviation of the sample of size k
    SEM = np.round(std / np.sqrt(len(df)), 4)
    w = 2*1.96*SEM
    print(str(mean) + "&" + str(std) + "&" + str(SEM) + "&" + str(w))

    return mean, std, SEM, w

def create_subsampling_data(K_samples, data, save_dir, name, metric_name):
    """
     # accumulate 100 statistical values on a subset of size K : {10, 20, 30, 50, 100}
     and stores them in a csv file of columns: sample-set	u-kj	sigma-kj	SEM-kj
     returns the absolute path to the csv file.

     The function also conducts Bootstrapping on the sub-samples and
     returns the values of mu-kj_star, sem-kj_star and w-kj_star

    """

    # Analytical set values
    SET_k = []
    
    SET_W_kj = []
    SET_u_k_j = []
    SET_sigma_k_j = []
    SET_SEM_k_j = []
    Set_GCI_0 = []
    Set_GCI_1 = []

    # Bootstrap set values
    SET_W_kj_star = []
    SET_u_star_k_j = []
    SET_sigma_star_k_j = []
    SET_SEM_star_k_j = []
    SET_BCI_0 = []
    SET_BCI_1 = []

    for k in K_samples:
        # SET_u_k_j set of means of the sample set of size k < N=110 for hippocampus
        for j in tqdm(range(0,100)): # the experiment needs to be repeated a 100 times since in small samples the values will highly depend on sampled data.

            dice_acccuracies_Skj = data['metric'].sample(k) #N samples were selected without replacement

            # Statistical Analysis of the sub samples
            # get the mean & standard deviation of the sample of size k

            mean_kj = np.mean(dice_acccuracies_Skj)
            std_kj = np.std(dice_acccuracies_Skj)
            sem_kj =  np.round(std_kj/ np.sqrt(k), 4)
            w_kj = 2*1.96*sem_kj
            GCI = [np.round(mean_kj - 1.96 * std_kj / np.sqrt(k), 4), np.round(mean_kj + 1.96 * std_kj / np.sqrt(k), 4)]

            SET_k.append(k) 
            SET_u_k_j.append(mean_kj) # list of means of 100 k samples
            SET_sigma_k_j.append(std_kj)  # list of standard deviation of 100 k samples
            SET_SEM_k_j.append(sem_kj) # list of SEM of a 100 k samples
            SET_W_kj.append(w_kj)
            Set_GCI_0.append(GCI[0])
            Set_GCI_1.append(GCI[1])


            # Bootstrapping Analysis of the sub-samples
            mean_kj_star, std_kj_star, w_kj_star, BCI = Bootstrap_Analysis(dice_acccuracies_Skj)

            SET_u_star_k_j.append(mean_kj_star)
            SET_SEM_star_k_j.append(std_kj_star)
            SET_W_kj_star.append(w_kj_star)
            SET_BCI_0.append(BCI[0])
            SET_BCI_1.append(BCI[1])

    sample_data = {'sample-set':SET_k, 'u-kj': SET_u_k_j,'sigma-kj': SET_sigma_k_j, 'SEM-kj': SET_SEM_k_j,
                   'w-kj': SET_W_kj, 'A-kj':Set_GCI_0, 'B-kj':Set_GCI_1 ,
                   'u-kj-star': SET_u_star_k_j,
                   'SEM-kj-star': SET_SEM_star_k_j,'w-kj-star': SET_W_kj_star, 'A-kj-star': SET_BCI_0,
                   'B-kj-star':SET_BCI_1}

    df = pd.DataFrame(sample_data)
    today = date.today()
    d4 = today.strftime("%b%d")
    #print(d4)
    file = "subsampled-stats-{}-{}-{}.csv".format(metric_name, name,str(d4))
    df.to_csv(os.path.join(save_dir,file))

    return os.path.join(save_dir,file)

def get_ecdf(data):
    # Get lenght of the data into n
    n = len(data)

    # We need to sort the data
    x = np.sort(data)

    # the function will show us cumulative percentages of corresponding data points
    y = np.arange(1, n + 1) / n

    return x, y

def draw_bs_replicates(data, func, size):
    """creates a bootstrap sample, computes replicates and returns replicates array
    data: the data to do bootstrap on.
    func: parameter corresponds to the summary statistics that we will want to use when create a bootstrap replicate.
    'size' parameter will demonstrate that how many replicates we need
    """
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)

    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        # replaceboolean, optional
        # Whether the sample is with or without replacement. Default is True, meaning that a value of a can be selected multiple times.
        bs_sample = np.random.choice(data, size=len(data)) #drawing a k sampled size with replacement
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)

    return bs_replicates

from pandas import Series
def Bootstrap_Analysis(data, k= None):
    ""
    """
    Conducts the Bootstrapping with bootstrap sample of size k, 
    if k is none takes the entire dataset, else it extracts without replacement a 
    sub-sample of size k and conducts bootstrapping on this sub-sample.
    
    In our code, we only use the functionality of K= None since the sub-samples are 
    predefined in the folder sub-sample. so we just give it the sub-sample instead of 
    the whole dataset.
    
    The function returns the bootstrap mean, standard error and confidence interval width:
    u_star, sem_star, and w_star to be stored in the csv file sub-sample
    """
    if type(data) == pd.core.series.Series:
        data = data.to_frame(name='metric')

    if k is None:
        k = len(data)

    dice_acccuracies = data['metric'].sample(k).reset_index(
        drop=True)  # k samples were selected without replacement when its k =110 samples for hippo or k = 334 for brain, the sampls set remains unchanged from the test set
    if k is None:
        assert sorted(data['metric']) == sorted(dice_acccuracies)
    # Reach in and draw out one slip, write that number down, and put the slip back into the bag. Repeat Step 2 as many times as needed to match the number of measurements you have, returning the slip to the bag each time.
    bs_replicates = draw_bs_replicates(dice_acccuracies, np.mean,
                                       15000)  # sample 15000 times from the subset of size n with replacement.

    # Get the corresponding values of 2.5th and 97.5th percentiles
    conf_interval = np.array(np.percentile(bs_replicates, [2.5, 97.5]).round(2))

    w_star = conf_interval[-1] - conf_interval[0]
    mu_star = np.mean(bs_replicates)
    sem_star = np.std(bs_replicates)
    #print("w*={}".format(conf_interval[-1] - conf_interval[0]))
    #print(str(mu_star) + "&" + str(sem_star) + "&" + str(w_star))
    return mu_star, sem_star, w_star, conf_interval

