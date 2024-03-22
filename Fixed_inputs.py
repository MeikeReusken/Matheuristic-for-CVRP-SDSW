import numpy as np

## -------------------------- Fixed inputs -----------------------##

import pandas as pd

rnd = np.random
rnd.seed(0)

s = 50  # specify speed: time = distance/speed, 50 kilometer per hour

# Safety factors
Gamma = 0.01  # safety factor for capacity to determine starting k
Delta = 0.01  # safety factor for time to determine starting k
Beta = 0.05  # safety factor for time in Assignment problem | in Reusken et al. (2023) this is denoted by Eta
Alpha = 0.05  # safety factor for capacity in Assignment problem

# Duration equality parameters | not described in Reusken et al. (2023)
# Enforces equality of expected travel times between districts: limit the difference to b_equity hours and relax when no feasible solution is found
# The coded matheuristic allows for turning this option on and off
b_equity = 2  # hours allowed between the durations
theta = 0.5  # duration equality relaxation over iterations

# Parameters for transformation by Dror et al. (1993)
inf = 10000
lam = 1000  # arbitrary large number to connect nodes n+1 and n+2 in the routing problem

# beta's for TSP approximations from Franceschetti et al. (2017)
Fran = {20: 0.8584265,
        30: 0.8269698,
        40: 0.8129900,
        50: 0.7994125,
        60: 0.7908632,
        70: 0.7817751,
        80: 0.7775367,
        90: 0.7773827,
        100: 0.7764689,
        110: 0.7764689 + (0.7563542 - 0.7764689) / 10}  # 0.7563542 is for n = 200
# interpolation to determine n+1's
Fran_inter = {i + j: Fran[i] + 1 / 10 * j * (Fran[i + 10] - Fran[i]) for i in [20, 30, 40, 50, 60, 70, 80, 90, 100] for
              j in range(10)}

# Print summary
print('-----FIXED INPUTS-----')
print(
    '\ns = ', s, 'km per hour',
    '\n\nSafety factors: ',
    '\nGamma = ', Gamma, 'for capacity to determine starting k',
    '\nDelta = ', Delta, 'for time to determine starting k',
    '\nAlpha = ', Alpha, 'for capacity in Assignment problem',
    '\nBeta = ', Beta, 'for time in Assignment problem')

Fixed_inputs = pd.DataFrame([Gamma, Delta, Beta, Alpha, s],
                            index=['Gamma', 'Delta', 'Beta', 'Alpha', 's'])

Fixed_inputs.to_excel('Output/Fixed_inputs.xlsx')
