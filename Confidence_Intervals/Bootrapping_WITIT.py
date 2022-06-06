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


def draw_bs_replicates(data, func, size, N):
    """creates a bootstrap sample, computes replicates and returns replicates array
    data: the data to do bootstrap on.
    func: parameter corresponds to the summary statistics that we will want to use when create a bootstrap replicate.
    'size' parameter will demonstrate that how many replicates we need usuallt 15000
    """
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)

    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data, size=N)
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)

    return bs_replicates

# Import the iris dataset
data = pd.read_csv('/Results/results_fold_3.csv')
data[' dice'] = (data[' dice']*100).round(2)
print("mean of the entire dataset is {}".format(data[' dice'].mean()))
print("std of the entire dataset is {}".format(data[' dice'].std()))
# Looking at information
print("\nInfo:")
print(data.info())

# Summary Statistics
print("\nSummary Statistics:")
print(data.describe())

# Display first 5 rows
print(data.head())
N_samples = [10,20,30,50,100,200,300,500,1000]
print("N-samples " + "&" + "Bootstrap method \\\\  ")
for n in N_samples:
    # Extract 500 random heights

    dice_acccuracies = data[' dice'].sample(n).reset_index(drop=True)

    # Display Summary Statistics of heights in cm
    # print(dice_acccuracies.describe())

    # Get Standard Deviation and Mean
    dice_std = np.std(dice_acccuracies)
    dice_mean = np.mean(dice_acccuracies)

    bs_replicates = draw_bs_replicates(data[' dice'],np.mean,15000, n)

    # Print the mean of bootstrap replicates
    #print( " Bootstrap replicates mean: " + str(np.mean(bs_replicates)))
    #print(" Bootstrap replicates std: " + str(np.std(bs_replicates)))
    #print("CV : Coefficient of variance is " + str(np.std(bs_replicates)/np.mean(bs_replicates) ))
    boot_std = np.std(bs_replicates)
    boot_mean = np.mean(bs_replicates)
    # Get the corresponding values of 2.5th and 97.5th percentiles
    conf_interval = np.percentile(bs_replicates,[2.5,97.5]).round(2)
    GCI = [np.round(boot_mean - 2 * boot_std / np.sqrt(n), 2), np.round(boot_mean + 2 * boot_std / np.sqrt(n), 2)]
    # Print the interval
    print(str(n) + "& " + str(conf_interval) + "&" + str((conf_interval[-1] - conf_interval[0]).round(2)) +
          "&" + str( GCI ) + "&" + str(np.round(boot_std/boot_mean, 2)))



