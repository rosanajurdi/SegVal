"""
Script to generate the normal (gaussian) distribution confidence interval values. and print them onto a latex table
the mu : mean is set to 60
the standard deviation to a range of values between 0.6 (smallest case) and 60  (highest illogical case)
CV remains the same
N samples between 10 and 1000.

"""
#mu = 60
#sigma = [0.6,1.8,3.6,18,28.8, 36, 54,57.6,59.4]

mu = 82.
sigma = [ 2.1 , 3.5 , 4.2 ,6.97,  7.7,  9.1]
N_samples = [10,20,30,50,100,200,300,500,1000]

import numpy as np
for s in sigma:
    print('{}&'.format(np.round(np.float(s)*100/mu, 2)))
# for a 95 confidence interval statistic the rule is X - 2*sigma/sqrt(n)
print('\hline')
for n in N_samples:
    print('$k = {}$'.format(n))
    for s in sigma:
        print( "&", [np.round(mu - 1.96*s/np.sqrt(n),2), np.round(mu + 1.96*s/np.sqrt(n), 2)])
    print('\\ \ ')

for k in N_samples:
    print('$k = {}$'.format(k))
    for s in sigma:
        print( "&", np.round(s/np.sqrt(k), 2),'&', np.round(2*1.96*s/np.sqrt(k), 2))
    print('\\ \ ')