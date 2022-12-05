# Import necessary libraries
"""
this script runs as input the path to the dataset and
runs an entire statistical analysis and Bootstrap analysis on
it. It is the script used to recreate the results in the paper.

input: just the csv file with the test set samples and their performance metric.

"""
import numpy as np
import pandas as pd
import seaborn as sns
from Confidence_Intervals.Statistical_functions import statistical_analysis, create_subsampling_data, Bootstrap_Analysis
import matplotlib.pyplot as plt
import os
from datetime import date, datetime
import argparse
from Confidence_Intervals.Generate_Gaussian_Dist_CV_Table import Generate_gaussian_dist_table

def run(args: argparse.Namespace):
    # Import the dataset
    # root_path = '/Users/rosana.eljurdi/PycharmProjects/SegVal_Project/Stats/Brain_Tumor/test/my_name.csv'
    name = args.dataset
    root_path = args.root_path

    visualize = args.visualize
    visualize_sabsamples = args.visualize_sabsamples

    K_samples = args.K_samples

    path = root_path.split(os.path.basename(root_path))
    data = pd.read_csv(root_path)
    if len(data.columns) == 2:
        data.columns = ['index', 'metric']
    elif len(data.columns) == 3:
        data.columns = ['id', 'index', 'metric']
        data = data[['index', 'metric']]



    if 'hauss' in os.path.basename(root_path):
        metric_name = 'HD'

        #data['metric'] *= 100
        #data['metric'] = (data['metric']).round(2)
    elif 'dice' or 'Dice' in os.path.basename(root_path):
        metric_name = 'DSC'


    # create the sub-sample file
    today = datetime.now()
    d4 = today.strftime("%b%d%H")
    folder = "subsampled-stats-{}-{}-{}.csv".format(metric_name, name, str(d4))
    print("results will be saved under ", folder)
    # print Analytical statistical alues:
    print("Analytical statistical values")
    mean, std, SEM, w, CI = statistical_analysis(data)

    # Generate the Gaussian Table:
    print("Generate the Gaussian Table")
    #Generate_gaussian_dist_table(std, path[0])

    print("Conduct Bootstrapping on the entire dataset")
    # Conduct Bootstrapping on the entire dataset.
    mu_star, sem_star, w_star, conf_interval = Bootstrap_Analysis(data)
    #print(str(str(w) + "star:" + str(w_star)))


    print("stats and bootstrapp: {}  &  {}  &  {}  &  [-{}, {}]  & {} &  {} &  [-{}, {}]".format(
        mean,std,SEM,-w / 2, w / 2,mu_star,sem_star,-w_star / 2.0, -w_star / 2.0
    ))


    #Visualize the Distribution of the entire dataset:
    if visualize is True:
        data['Dice'] = data['metric']
        data.boxplot(column='Dice')
        plt.title('Box Plot of {} Dataset.'.format(name))
        plt.savefig(os.path.join(path[0], 'Boxplot-{}-{}'.format(name, metric_name)))

        plt.figure()
        sns.distplot(data['Dice'], hist=True, kde=True,
                     bins=int(180 / 5), color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 4})

        # plt.legend(prop={'size': 16}, title='Hippo-Distribution')
        plt.title('Histogram and Density Distribution of {} Dataset.'.format(name))
        plt.savefig(os.path.join(path[0], 'Distribution-{}-{}'.format(name, metric_name)))



    save_dir = os.path.join(path[0],folder.split('.csv')[0])
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)


    a = create_subsampling_data(K_samples, data, save_dir, name, metric_name)
    root_path = a
    basepath = root_path.split(os.path.basename(root_path))[0]
    df = pd.read_csv(root_path)
    '''
    if visualize_sabsamples is True:
        # Visualizing the subsampled distribution
        df.boxplot(column='u-kj', by='sample-set')
        plt.savefig(os.path.join(basepath, "Boxplot-{}-{}-u-kj".format(metric_name, name)))
        #df.hist(column='u-kj', by='sample-set')
        #plt.savefig(os.path.join(basepath, "Hist-Hippo-u-kj"))
        print("hello")

        df.boxplot(column='sigma-kj', by='sample-set')
        plt.savefig(os.path.join(basepath, "Boxplot-{}-{}-sigma-kj".format(metric_name,name)))
        #df.hist(column='sigma-kj', by='sample-set')
        #plt.savefig(os.path.join(basepath, "Hist-Hippo-sigma-kj"))
        print('hello 2')

        df.boxplot(column='SEM-kj', by='sample-set')
        plt.savefig(os.path.join(basepath, "Boxplot-{}-{}-SEM-kj".format(metric_name,name)))
        #df.hist(column='SEM-kj', by='sample-set')
        #plt.savefig(os.path.join(basepath, "Hist-Hippo-SEM-kj"))
        print('hello 2')
    '''
    print("Statistical Analysis ")
    txt_file= open(os.path.join(save_dir, 'txt_log.txt'), 'w')
    print("Bootstrapping Analysis ")
    txt_file_Boot = open(os.path.join(save_dir, 'Boot_txt_log.txt'), 'w')

    for k in K_samples:
        df_sample = df[df['sample-set']== k]
        assert df_sample.__len__() == 100 # Make sure that there are only a 100 samples selected.

        mean_variation = '{} $\pm$ {} '.format(df_sample['u-kj'].mean().round(3), df_sample['u-kj'].std().round(3))
        std_variation = '{} $\pm$ {}'.format(df_sample['sigma-kj'].mean().round(3), df_sample['sigma-kj'].std().round(3))
        SEM_Variation =  '{} $\pm$ {}'.format(df_sample['SEM-kj'].mean().round(3), df_sample['SEM-kj'].std().round(3))
        Width_Variation = '{} $\pm$ {}'.format(df_sample['w-kj'].mean().round(3), df_sample['w-kj'].std().round(3))
        width_k = df_sample['w-kj'].mean().round(2)
        CI = '[-{}, {}]'.format(width_k/2.0, width_k/2)

        txt_file.write('$k =  {{k}}$ & ' + '\n ')




        mean_variation_star = '{} $\pm$ {} '.format(df_sample['u-kj-star'].mean().round(3),
                                                    df_sample['u-kj-star'].std().round(3))
        SEM_Variation_star =  '{} $\pm$ {}'.format(df_sample['SEM-kj-star'].mean().round(3),
                                                   df_sample['SEM-kj-star'].std().round(3))
        Width_Variation_star = '{} \pm {}'.format(df_sample['w-kj-star'].mean().round(3),
                                                  df_sample['w-kj-star'].std().round(3))
        CI_star = '[-{}, {}]'.format(df_sample['w-kj-star'].mean().round(3) / 2.0, df_sample['w-kj-star'].mean().round(3) / 2)
        txt_file_Boot.write('$k$ = ' + str(k) + "$&$" +mean_variation_star +  "$&$" + SEM_Variation_star + "$&$"
                            + "$&$"  + Width_Variation_star + "$&$"   + "$"  + '\n ')

        print('$k$ = ' + str(k) + "&" + mean_variation + "&" + std_variation + "&" + SEM_Variation +
                       "&" + CI + "&"  +mean_variation_star +  "&" + SEM_Variation_star + "&" + CI_star)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, default='hippo')

    parser.add_argument('--root_path', type=str,
                        default='/Users/rosana.eljurdi/PycharmProjects/nnUNet_SegVal/nnUNet_preprocessed/Task001_SegVal_BrainTumor/results-hauss-3D-L1.csv')

    parser.add_argument('--visualize' ,  type=str, default=True)
    parser.add_argument('--visualize_sabsamples', type=str, default=True)
    parser.add_argument('--K_samples', type=list,
                        default=[10, 20, 30, 50, 100, 150, 200,250,300,334])

    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    #Guanghui_test_set = [73.96,66.31,61.93,75.96, 58.92,76.26,72.39,77.81,78.55,67.00,53,43,70,80,62,80]

    #data = pd.DataFrame(data=Guanghui_test_set, columns=['metric'])
    #data['metric'] = data['metric']/100
    #a = statistical_analysis(data)
    #print(a)
    #b = Bootstrap_Analysis(data)
    #print(b)
    #run(get_args())
    Generate_gaussian_dist_table()


