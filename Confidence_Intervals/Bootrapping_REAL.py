# Import necessary libraries
import numpy as np
import pandas as pd


# Create a function to get x, y for of ecdf
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
        bs_sample = np.random.choice(data, size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)

    return bs_replicates
import os
# Import the iris dataset
root_path = '/Results/Hippo/csv/results-hauss.csv'
data = pd.read_csv(root_path)
if 'dice' in os.path.basename(root_path) :
    data[' metric '] *=100

data[' metric '] = (data[' metric ']).round(2)
print("mean of the entire dataset is {}".format(data[' metric '].mean()))
print("mean of the entire dataset is {}".format(data[' metric '].std()))

# Looking at information
print("\nInfo:")
print(data.info())

# Summary Statistics
print("\nSummary Statistics:")
print(data.describe())

# Display first 5 rows
print(data.head())
N_samples = [10,20,30,50,100,110]
print("N-samples " + "&" + "Bootstrap method \\\\  ")
from tqdm import tqdm
boot_std = 0
boot_mean = 0
for n in N_samples:
    # Extract 500 random heights
    conf_interval = np.zeros((2))
    se = 0
    for i in tqdm(range(0,100)):
        dice_acccuracies = data[' metric '].sample(n).reset_index(drop=True)
        # Display Summary Statistics of heights in cm
        # print(dice_acccuracies.describe())
        # Get Standard Deviation and Mean
        dice_std = np.std(data[' metric '])
        dice_mean = np.mean(data[' metric '])
        bs_replicates = draw_bs_replicates(dice_acccuracies,np.mean,15000)
        # print("Empirical mean: " + str(dice_mean))
        # Print the mean of bootstrap replicates
        # print( " Bootstrap replicates mean: " + str(np.mean(bs_replicates)))
        # print(" Bootstrap replicates std: " + str(np.std(bs_replicates)))
        boot_std += np.std(bs_replicates)
        boot_mean += np.mean(bs_replicates)
        # Get the corresponding values of 2.5th and 97.5th percentiles
        conf_interval += np.array(np.percentile(bs_replicates, [2.5, 97.5]).round(2))
        se += np.std(bs_replicates) / np.mean(bs_replicates)
        # Print the interval
    boot_std = boot_std/100.00
    boot_mean = boot_mean/100.00
    GCI = [np.round(boot_mean - 2 * boot_std / np.sqrt(n), 2), np.round(boot_mean + 2 * boot_std / np.sqrt(n), 2)]
    print(str(n) + "& " + str(conf_interval/100.00) + "&" + str((conf_interval[-1]/100.00 - conf_interval[0]/100.00).round(2)) +
          "&" + str(GCI) + "&" + str(np.round(boot_std / boot_mean, 2)))



