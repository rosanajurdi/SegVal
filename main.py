# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from Confidence_Intervals.Statistical_functions import statistical_analysis, create_subsampling_data
import matplotlib.pyplot as plt
import os
# Import the dataset
root_path = '/Users/rosana.eljurdi/PycharmProjects/SegVal_Project/Stats/Test/fold_3/results-dice-3D.csv'
data = pd.read_csv(root_path)
# Create the pandas dataframe for the statistical experiments 

# print Analytical statistical alues:
statistical_analysis(data)

data.columns = ['file', 'metric']
if 'dice' in os.path.basename(root_path) :
    data['metric'] *= 100
    data['metric'] = (data['metric']).round(2)


K_samples = [ 10,20,30,50,100,110 ]
save_dir = root_path.split(os.path.basename(root_path))[0]

a = create_subsampling_data(K_samples, data, save_dir)
# root_path = "/Users/rosana.eljurdi/PycharmProjects/SegVal_Project/Stats/Test/fold_3/subsampled-stats-Hippo-Sep16.csv"
root_path = a
basepath = root_path.split(os.path.basename(root_path))[0]
df = pd.read_csv(root_path)


#
df.boxplot(column='u-kj', by='sample-set')
plt.savefig(os.path.join(basepath, "Boxplot-Hippo-u-kj"))
df.hist(column='u-kj', by='sample-set')
plt.savefig(os.path.join(basepath, "Hist-Hippo-u-kj"))
print("hello")

df.boxplot(column='sigma-kj', by='sample-set')
plt.savefig(os.path.join(basepath, "Boxplot-Hippo-sigma-kj"))
df.hist(column='sigma-kj', by='sample-set')
plt.savefig(os.path.join(basepath, "Hist-Hippo-sigma-kj"))
print('hello 2')

df.boxplot(column='SEM-kj', by='sample-set')
plt.savefig(os.path.join(basepath, "Boxplot-Hippo-SEM-kj"))
df.hist(column='SEM-kj', by='sample-set')
plt.savefig(os.path.join(basepath, "Hist-Hippo-SEM-kj"))
print('hello 2')

for k in K_samples:
    df_sample = df[df['sample-set']== k]
    assert df_sample.__len__() == 100 # Make sure that there are only a 100 samples selected.

    mean_variation = '{} \pm {}'.format(df_sample['u-kj'].mean().round(2), df_sample['u-kj'].std().round(2))
    std_variation = '{} \pm {}'.format(df_sample['sigma-kj'].mean().round(2), df_sample['sigma-kj'].std().round(2))
    SEM_Variation =  '{} \pm {}'.format(df_sample['SEM-kj'].mean().round(2), df_sample['SEM-kj'].std().round(2))
    plt.figure()
    sns.distplot(df_sample['u-kj'], hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.legend(prop={'size': 16}, title='Hippo-{}-u-kj'.format(N))
    plt.title('Histogram and Density Distribution of Mean Dice of sample N = {}'.format(N))


    plt.savefig(os.path.join(basepath, 'Distribution-Hippo-{}'.format(N)))

    print('k = ' + str(N) + "&" +mean_variation + "&" + std_variation + "&" + SEM_Variation)



       
    




