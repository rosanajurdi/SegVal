"""
Script to generate the normal (gaussian) distribution confidence interval values. and print them onto a latex table
the mu : mean is set to 60
the standard deviation to a range of values between 0.6 (smallest case) and 60  (highest illogical case)
CV remains the same
N samples between 10 and 1000.

"""
#mu = 60
#sigma = [0.6,1.8,3.6,18,28.8, 36, 54,57.6,59.4]

"""
Script to generate the normal (gaussian) distribution confidence interval values. and print them onto a latex table
the mu : mean is set to 60
the standard deviation to a range of values between 0.6 (smallest case) and 60  (highest illogical case)
CV remains the same
N samples between 10 and 1000.

"""

import os
import numpy as np
def Generate_gaussian_dist_table(sd, save_dir):
    N_samples = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, 2000]
    sigma = [2, 5, 8, 10.45, sd, 12, 15, 18]
    if save_dir is not None:
        txt_file = open(os.path.join(save_dir, 'Gaussian_table_{}.txt'.format(sd)), 'w')



    # for a 95 confidence interval statistic the rule is X - 2*sigma/sqrt(n)
    print('\hline')

    '''
    for n in N_samples:
        print('$k = {}$'.format(n))
        for s in sigma:
            print( "&", [np.round(mu - 1.96*s/np.sqrt(n),2), np.round(mu + 1.96*s/np.sqrt(n), 2)])
        print('\\ \ ')
    '''

    txt_file.write('sigma = ' + str(sigma) )
    print('sigma = ' + str(sigma) )
    for k in N_samples:
        print('$k = {}$'.format(k))
        if save_dir is not None:
            txt_file.write('$k = {}$'.format(k))
        for s in sigma:
            print( "&", np.round(s/np.sqrt(k), 2),'&', np.round(2*1.96*s/np.sqrt(k), 2))
            if save_dir is not None:
                txt_file.write("&" + str(np.round(s/np.sqrt(k), 2))+'&'+ str(np.round(2*1.96*s/np.sqrt(k), 2))  )

                txt_file.write("\\ \ " )
'''
if __name__ == '__main__':
    Generate_gaussian_dist_table(80.70, 10.75, None )
'''