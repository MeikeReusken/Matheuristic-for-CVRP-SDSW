# some of the below packages are only used when using a specific setting of the matheuristic
import math
import numpy as np
import scipy.stats as st
from scipy.stats import poisson
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.heuristics import solve_tsp_local_search
from math import atan2, degrees, atan
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt
import time
from Fixed_inputs import *
from itertools import cycle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import geopy.distance


# --------------------------- Get data ------------------------ #
def Data(track_data_set, a, xc, yc, n, Plot):
    rnd = np.random
    rnd.seed(0)

    N = [i for i in range(1, n + 1)]  # set of customers
    V = [0] + N  # set of nodes, where "0" denotes the depot

    if track_data_set in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
        print('Calculate cost matrix using geopy to obtain km distance from lon and lat')
        c = np.zeros((1 + n, 1 + n))  # 1 for the depot, n for the number of customers
        for i in V:
            for j in V:
                if i != j:
                    c[i, j] = geopy.distance.geodesic((xc[i], yc[i]), (xc[j], yc[j])).km

    if track_data_set not in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
        c = np.zeros((1 + n, 1 + n))  # 1 for the depot, n for the number of customers
        for i in V:
            for j in V:
                if i != j:
                    c[i, j] = np.hypot(xc[i] - xc[j], yc[i] - yc[j])

    # probabilities of waiting
    p_w_groups = rnd.randint(0, 4, n)  # randomly assign each customer to group 0 to group 3
    p_w_input = [[(0.1, 0.02), (0.05, 0.02), (0.005, 0.002)],  # (mean, std) for each group
                 [(0.2, 0.05), (0.1, 0.05), (0.05, 0.05)],
                 [(0.01, 0.01), (0.001, 0.001), (0, 0)],
                 [(0.15, 0.1), (0.002, 0.1), (0.001, 0.1)]]
    p_w = [[rnd.normal(p_w_input[i][j][0], p_w_input[i][j][1], 1)[0] for j in [0, 1, 2]] for i in p_w_groups]
    p_w = np.clip(p_w, 0, 1)  # truncate at 0 (this also changes p_w from matrix to np.array)

    b = [a * (2 + 0.5 * p_w[i - 1][0] + 1.5 * p_w[i - 1][1] + 2.5 * p_w[i - 1][2]) for i in
         N]

    if Plot == 'yes':
        plt.plot(xc[0], yc[0], c='r', marker='s')
        plt.scatter(xc[1:], yc[1:], c='b')
        plt.show()

    # print('Main cost matrix: ', c)

    return p_w, N, V, c, b


# --------------------------- Determining the number of districts ------------------------ #
def starting_k(track_data_set, xc, yc, Q, c, b, mu, sd, T, n, method):
    print('\n----- DETERMINE NUMBER OF DISTRICTS k -----')
    print('Daily time limit in hours: ', T)
    reason_k = []
    # the probability of meeting the total vehicles capacity should be at least Gamma
    k_capacity = (sum(mu) + st.norm.ppf(1 - Gamma) * np.sqrt(sum(sd ** 2))) / Q

    # the probability of meeting the travel time limit should be at least Delta
    # duration of recourse and driving is unknown and we tested two methods for approximation: 'solve_TSP' and 'approx_TSP_Beardwood'
    # METHOD 1: solve TSP
    if method == 'solve_TSP':
        if n < 20:
            permutations, distance = solve_tsp_dynamic_programming(c)
        if n >= 20:
            permutations, distance = solve_tsp_local_search(c)
        d_approx = distance / s
    # METHOD 2: Approximate TSP route taking into account that there are m vehicles
    if method == 'approx_TSP_Beardwood':
        if track_data_set in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
            max_x = np.argmax(xc)
            min_x = np.argmin(xc)
            max_y = np.argmax(yc)
            min_y = np.argmin(yc)
            area = geopy.distance.geodesic((xc[max_x], yc[max_x]), (xc[min_x], yc[min_x])).km * geopy.distance.geodesic(
                (xc[max_y], yc[max_y]), (xc[min_y], yc[min_y])).km
        if track_data_set not in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
            area = (max(xc) - min(xc)) * (max(yc) - min(yc))
        if n >= 20:
            d_approx = (Fran_inter[n + 1] * np.sqrt(area * (n + 1))) / s
        if n < 20:
            permutations, distance = solve_tsp_local_search(c)
            d_approx = distance / s  # take tsp solution as an approximation

    print('\nApproximation 1: ', d_approx)

    k_time = (d_approx + sum(b * mu) + st.norm.ppf(1 - Delta) * np.sqrt(sum((np.array(b) ** 2) * (sd ** 2)))) / T

    # take largest k
    k = max(math.ceil(k_capacity), math.ceil(k_time))
    K = [i for i in range(1, k + 1)]

    if math.ceil(k_capacity) > math.ceil(k_time):
        reason_k = '(c)'
    if math.ceil(k_capacity) < math.ceil(k_time):
        reason_k = '(t)'
    if math.ceil(k_capacity) == math.ceil(k_time):
        reason_k = '(t,c)'

    print('Minimum k to meet capacity constraint = ', k_capacity)
    print('Minimum k to meet time constraint = ', k_time)
    print('Starting k = ', k)

    return k, K, reason_k, d_approx


# --------------------------- Seed selection ------------------------ #

def Seeds(N, c, k, K, xc, yc, n, d, method):
    print('\n-----DETERMINE SEEDS N_s-----')

    if k <= 1:
        print('!!Only  one cluster, skip some actions')
        cone_allocation = [] + [range(n)]
        # print(cone_allocation)

    else:
        def get_angle(point_1, point_2):
            angle = atan2(point_1[1] - point_2[1],
                          point_1[0] - point_2[0])  # use atan2 instead of atan as the depot is the origin
            angle = degrees(angle)
            return angle

        angles = [get_angle([xc[0], yc[0]], [xc[i], yc[i]]) for i in N]
        for i in N:
            if angles[i - 1] < 0:
                angles[i - 1] = 360 + angles[i - 1]

        # Three methods for seed selection were tested: 'Fisher', 'Sultana', 'KMeans'
        if method == 'Fisher':
            # proposed by Fisher and Jaikumar (1981)
            # does not work when there is an empty cone
            print('\nApproach 1: Fisher and Jaikumar (1981)')

            # size of each cone:
            cone_size = (max(angles) - min(angles)) / k  # gives degrees of cone

            # create set N_k, where N_1,..,N_k give the set of customer allocation in that cone
            # start division from the node with the smallest angle
            cone_allocation = []
            for k in K:
                cash = []
                for i in N:
                    if (min(angles) + (k - 1) * cone_size) <= angles[i - 1] <= (min(angles) + k * cone_size):
                        cash = cash + [i]
                cone_allocation = cone_allocation + [cash]

        if method == 'Sultana':
            # proposed by Sultana et al. (2017)
            # idea: divide the plane by the number of vehicle so that each cone holds equal number of nodes
            print('\nApproach 2: Sultana et al. (2017)')

            div, up = divmod(n, k)  # div gives n/k rounded down, up gives how many times should be rounded up
            cone_size = [div] * (k - up) + [div + 1] * up  # gives number of customers in a cone

            # start division from the node with the smallest angle
            # the below code splits starting at degree 0, this may not always be logical, so in case using Sultana consider first looking at the largest split and start from there 
            angles_sort = sorted(angles)
            cone_size_cum = np.cumsum(cone_size)
            boundaries = [-1] + [angles_sort[i - 1] for i in cone_size_cum]
            a2 = np.array(angles)

            cone_allocation = []
            for i in range(len(boundaries) - 1):
                cash = set(np.where(a2 <= boundaries[i + 1])[0]).intersection(np.where(a2 > boundaries[i])[0])
                cone_allocation = cone_allocation + [list(cash)]

        if method == 'KMeans':
            print('\nApproach 3: K-Means clustering')
            km = KMeans(n_clusters=k)
            cluster = pd.DataFrame(np.transpose([xc[1:], yc[1:]]))
            y_predicted = km.fit_predict(cluster)
            cluster['cluster'] = y_predicted
            # print(cluster)
            cone_allocation = []
            for i in range(k):
                cash = cluster[cluster['cluster'] == i].index
                cone_allocation = cone_allocation + [list(cash)]

        print('Customer cones : ', cone_allocation)

    # Pick a seed customer in each cone = SAME FOR ALL METHODS
    new_cone_allocation = [[x + 1 for x in cone_allocation[i]] for i in range(k)]
    # cone_allocation starts from 0 and new_cone_allocation starts from 1
    N_s_1 = []
    for k in K:
        if new_cone_allocation[k - 1] == []:
            N_s_1 = N_s_1 + ['NaN']  # only occurs when method = 'Fisher'
        if new_cone_allocation[k - 1] != []:
            for i in new_cone_allocation[k - 1]:
                if c[0][i] == max(c[0][new_cone_allocation[k - 1]]):
                    N_s_1 = N_s_1 + [i]
    N_s = [N_s_1[i] - 1 for i in range(len(N_s_1))]  # N_s starts from 0, N_s_1 starts from 1
    print('The seed customer are: N_s = ', N_s_1)

    # store coordinates of seeds
    xk = xc[N_s_1]
    yk = yc[N_s_1]

    # show clusters & seeds
    cycol = cycle('bgrcmyk')
    plt.figure()
    for l in range(k):
        plt.scatter(xc[new_cone_allocation[l]], yc[new_cone_allocation[l]], c=next(cycol))
    plt.plot(xc[0], yc[0], c='black', marker='s')
    plt.scatter(xk, yk, c='white', marker='*')
    plt.title(str(d) + ", n = " + str(n) + ", k =" + str(k))

    return N_s, N_s_1, cone_allocation, xk, yk


# --------------------------- Assignment of customers to districts ------------------------ #
def Clustering(track_data_set, Q, N, c, b, k, K, N_s_1, xk, yk,
               xc, yc, n, mu, sd, T, it, Capacity_type, Equity, Violated, previous, Eta, Reduce_variables,
               m_reason):
    print('\n-----ASSIGNMENT OF CUSTOMERS TO DISTRICTS-----')

    Assignment_status = 'feasible'
    Districts = ""
    Districts_0 = ""
    Clustering_objective = ""
    Clustering_gap = ""
    Clustering_solution = ""
    Clustering_all = ""

    N_cluster = list(set(N) - set(N_s_1))  # remove N_s from N
    A = [(i, k) for i in N for k in K]  # set of arcs between customers and seeds
    matrix_K = [(t, k) for t in K for k in K]  # connections among districts

    # print(N_s_1, K, k)

    # seed in each cluster
    seeds = [(N_s_1[i], K[i]) for i in range(k)]

    # calculate cost of direct travel from node i to seed k
    d_ik = np.zeros((n, k))
    if track_data_set in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
        # Calculate cost matrix using geopy to obtain km distance from lon and lat
        for i in N:
            for k in K:
                d_ik[i - 1, k - 1] = geopy.distance.geodesic((xc[i], yc[i]), (xk[k - 1], yk[k - 1])).km
        d_0i = [geopy.distance.geodesic((xc[0], yc[0]), (xc[i], yc[i])).km for i in N]
        d_0k = [geopy.distance.geodesic((xc[0], yc[0]), (xk[k - 1], yk[k - 1])).km for k in K]
        # print('d_ik = ', d_ik)
        # print('d_0i = ', d_0i)
        # print('d_0k = ', d_0k)

    if track_data_set not in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
        for i in N:
            for k in K:
                d_ik[i - 1, k - 1] = np.hypot(xc[i] - xk[k - 1], yc[i] - yk[k - 1])
        d_0i = [np.hypot(xc[0] - xc[i], yc[0] - yc[i]) for i in N]
        d_0k = [np.hypot(xc[0] - xk[k - 1], yc[0] - yk[k - 1]) for k in K]

    # calculate insertion cost (see Fisher and Jaikumar (1981))
    C = np.zeros((n, k))
    for i in N:
        for k in K:
            C[i - 1, k - 1] = d_0i[i - 1] + d_ik[i - 1, k - 1] - d_0k[k - 1]
    # print(C)

    if n > 50:
        if Reduce_variables == 'yes':
            par_percentage = 0.3  # indicates the fraction of the total number of districts that are set to 0
            Furthest = {}
            for i in N:
                # wrt insertion costs:
                Furthest[i] = np.argpartition(C[i - 1], math.trunc(par_percentage * k * (-1)))[
                              math.trunc(par_percentage * k * (-1)):] + 1
            # print(Furthest)

    # Preparations for artificial variable y used for linear reformulation of capacity chance constraint
    B = [(i, j, k) for i in N for j in N for k in
         K]  # to reduce number of additional variables y, reduce this matrix to include only j>i
    B_ij = [(i, j) for i in N for j in N]

    # Probabilistic Generalized Assignment problem #
    # GAP introduces by Fisher and Jaikumar (1981) extended with probabilistic constraints
    mdl = Model('Clustering')
    x = mdl.addVars(A, vtype=GRB.BINARY)

    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[i, k] * C[i - 1, k - 1] for i, k in A))

    mdl.addConstrs(x[i, k] == 1 for i, k in seeds)
    mdl.addConstrs(quicksum(x[i, k] for k in K) == 1 for i in N)

    # chance constraints on Q | two formulations are tested
    if Capacity_type == 'linear':
        y = mdl.addVars(B, vtype=GRB.BINARY)  # in Reusken et al. (2023) this is denoted z_ijk
        mdl.addConstrs(st.norm.ppf(1 - Alpha) ** 2 * quicksum(sd[i - 1] ** 2 * x[i, k] for i in N) + 2 * Q * quicksum(
            mu[i - 1] * x[i, k] for i in N) - quicksum(mu[i - 1] ** 2 * x[i, k] for i in N) - 2 * quicksum(
            mu[i - 1] * mu[j - 1] * y[i, j, k] for i, j in B_ij) <= Q ** 2 for k in K)
        mdl.addConstrs(y[i, j, k] >= x[i, k] + x[j, k] - 1 for i, j, k in B)
        mdl.addConstrs(y[i, j, k] <= x[i, k] for i, j, k in B)
        mdl.addConstrs(y[i, j, k] <= x[j, k] for i, j, k in B)
        mdl.addConstrs(Q - quicksum(mu[i - 1] * x[i, k] for i in N) >= 0 for k in K)

    if Capacity_type == 'convex':
        mdl.addConstrs(st.norm.ppf(1 - Alpha) ** 2 * quicksum(sd[i - 1] ** 2 * x[i, k] for i in N) + 2 * Q * quicksum(
            mu[i - 1] * x[i, k] for i in N) - quicksum(mu[i - 1] * x[i, k] for i in N) ** 2 <= Q ** 2 for k in K)
        mdl.addConstrs(Q - quicksum(mu[i - 1] * x[i, k] for i in N) >= 0 for k in K)
        mdl.update()

    # chance constraints on T : convex quadratic formulation
    mdl.addConstrs(
        st.norm.ppf(1 - Beta) ** 2 * quicksum(sd[i - 1] ** 2 * x[i, k] * b[i - 1] ** 2 for i in N) + 2 * (
                T - 2 * c[0][N_s_1[k - 1]] / s) * quicksum(
            (C[i - 1, k - 1] / s + mu[i - 1] * b[i - 1]) * x[i, k] for i in N) - quicksum(
            (C[i - 1, k - 1] / s + mu[i - 1] * b[i - 1]) * x[i, k] for i in
            N) ** 2 <= (T - 2 * c[0][N_s_1[k - 1]] / s) ** 2 for k in K)
    mdl.addConstrs(
        (T - 2 * c[0][N_s_1[k - 1]] / s) - quicksum((C[i - 1, k - 1] / s + mu[i - 1] * b[i - 1]) * x[i, k] for i in
                                                    N) >= 0 for k in K)

    # duration equality constraints
    if Equity == 'yes':
        z = mdl.addVars(K, vtype=GRB.CONTINUOUS)
        mdl.addConstrs(
            2 * c[0][N_s_1[k - 1]] / s + quicksum((C[i - 1, k - 1] / s + b[i - 1] * mu[i - 1]) * x[i, k] for i in N) ==
            z[k]
            for k
            in K)
        mdl.addConstrs(z[k] - z[t] <= Eta for k in K for t in K if k != t)
        mdl.addConstrs(z[t] - z[k] <= Eta for k in K for t in K if k != t)
        mdl.update()

    if Violated != []:
        mdl.addConstrs(
            quicksum(x[i, k] for i in Violated[l]) <= len(Violated[l]) - 1 for k in K for l in range(len(Violated)))

    if n > 50:
        if Reduce_variables == 'yes':
            mdl.addConstrs(x[i, k] == 0 for i in N for k in list(Furthest[i]))

    # time limit
    mdl.Params.TimeLimit = 3600  # If a certain limit in time (seconds) is reached, the optimizer will stop

    if n > 50:
        mdl.Params.MIPGap = 0.05

    mdl.Params.Threads = 96

    # https://support.gurobi.com/hc/en-us/community/posts/360071928312-Stopping-the-program-if-best-objective-has-not-changed-after-a-while
    def cb(model, where):

        if where == GRB.Callback.MIPNODE:
            # Get model objective
            obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

            # Has objective changed?
            if abs(obj - model._cur_obj) > 1e-8:
                # If so, update incumbent and time
                model._cur_obj = obj
                model._time = time.time()

            # Terminate if objective has not improved in 60 * 8 seconds
            if (time.time() - model._time > (60 * 8)) and (float(obj) != float(1e+100)):
                model.terminate()

    # Last updated objective and time
    mdl._cur_obj = float("inf")
    mdl._time = time.time()

    mdl.optimize(callback=cb)
    print(mdl.getAttr("Status"))

    # page 702, 598
    # https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/8.1/refman.pdf
    if mdl.getAttr("Status") == 3:
        Assignment_status = 'infeasible'
    elif (mdl.getAttr("Status") == 9) and (float(mdl._cur_obj) == float(1e+100)):
        Assignment_status = 'no_sol_in_time'

    else:
        Clustering_objective = mdl.getObjective().getValue()
        Clustering_gap = mdl.MIPGap
        Clustering_solution = mdl.X[0:n * k]
        Clustering_all = mdl.X[:]

        # PLOT #
        active_arcs = [a for a in A if x[a].x > 0.99]

        plt.figure()
        for i, k in active_arcs:
            plt.plot([xc[i], xk[k - 1]], [yc[i], yk[k - 1]], c='black', zorder=0)
        plt.scatter(xc, yc, c='black')
        plt.scatter(xk, yk, c='white', marker='*')
        plt.plot(xc[0], yc[0], c='black', marker='s')
        plt.title('Assignment, Iteratie ' + str(it))
        # plt.show()

        # District lists
        Districts = []
        for k in K:
            cash = []
            for i in range(len(active_arcs)):
                if active_arcs[i][1] == k:
                    cash = cash + [active_arcs[i][0]]
            Districts = Districts + [cash]
        print('\nDistricts N_1, ..., N_k are : ', Districts)

        # Districts starts from 1, Districts_0 starts from 0
        Districts_0 = []
        for i in range(k):
            Districts_0 = Districts_0 + [[p - 1 for p in Districts[i]]]

    return Districts, Districts_0, Clustering_objective, Clustering_gap, Clustering_solution, Assignment_status, Clustering_all


# --------------------------- Routing ------------------------ #

#  Transformation proposed by Dror et al. (1993), this function is used in Routing()
def Dror(track_data_set, xc, yc, N, Summary):
    V = [0] + N  # V includes [0], N excludes [0]
    n = len(N)

    xc = np.append(xc[V], [xc[0], xc[0]])  # copy the coordinates of the depot for node n+1 and n+2
    yc = np.append(yc[V], [yc[0], yc[0]])

    c = np.zeros(
        (1 + n + 2, 1 + n + 2))  # 1 for the depot, n for the number of customers and 2 for the artificial nodes

    if track_data_set in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
        print('Calculate cost matrix using geopy to obtain km distance from lon and lat')
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    c[i, j] = geopy.distance.geodesic((xc[i], yc[i]), (xc[j], yc[j])).km

    if track_data_set not in ['Moisson_Montreal', 'Eemsdelta', 'Zevenaar']:
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    c[i, j] = np.hypot(xc[i] - xc[j], yc[i] - yc[j])

    # Dror transformation Section 1
    for i in range(n + 1):
        c[i, n + 1] = c[i, 0]
        c[i, n + 2] = c[i, 0] + p * (c[i, 0] + c[0, i])

    c[0, n + 1] = inf
    c[0, n + 2] = inf

    # Dror transformation Section 2
    c[n + 1, n + 2] = -lam
    c[n + 2, n + 1] = -lam

    for j in range(n + 1):
        c[n + 1, j] = inf
        c[n + 2, j] = c[0, j]

    c[n + 1, 0] = 0
    c[n + 2, 0] = inf

    # print('\nCost for DROR route: ', c)

    # solve TSP
    permutation, distance = solve_tsp_dynamic_programming(
        c)  # =Exact, only possible for `small` districts, for the districts in our paper this was always small enough
    # if districts are larger, consider heuristics provided by python-tsp, see https://github.com/fillipe-gsm/python-tsp

    route = []
    route_ar = []
    for i in permutation:
        if i in range(n + 1):
            route = route + [V[i]]
            route_ar = route_ar + [V[i]]
        if i == n + 1:
            route_ar = route_ar + ['ar1']
        if i == n + 2:
            route_ar = route_ar + ['ar2']

    # Valuation of solution:
    cost = distance + lam
    driving_time = cost / s

    if Summary == 'yes':
        print('Permutations = ', permutation,
              '\nRoute = ', route_ar,
              '\nAdjusted cost = ', distance + lam)

    print('\nPermutations = ', permutation)
    if permutation.index(n + 2) < permutation.index(n + 1):
        policy = 'risk cost of a return trip'
        print('last customer = ', permutation[n])
        print('N :', N)
        print('real index last customer = ', N[permutation[n] - 1])
        policy_time = p * (c[permutation[n], 0] + c[0, permutation[n]]) / s

    if permutation.index(n + 2) > permutation.index(n + 1):
        policy = 'preventive break'
        policy_time = 0

    print('policy time = ', policy_time)
    print('driving time = ', driving_time)

    driving_only_time = driving_time - policy_time

    return permutation, distance, route, route_ar, policy, cost, driving_time, policy_time, driving_only_time


def Routing(track_data_set, a, p_w, k, Districts, Districts_0,
            xc, yc, mu, it, Summary):
    print('\n-----ROUTING-----')
    # Compute route for each district #

    Routes, Policies, Costs = [], [], []
    driving, driving_only, recourse = [], [], []
    Duration = pd.DataFrame(columns=['driving', 'handling', 'waiting', 'driving_only', 'recourse'])
    for i in range(k):
        permutations, distance, route, route_ar, policy, cost, driving_time, policy_time, driving_only_time = Dror(
            track_data_set, xc, yc, Districts[i], Summary='no')
        Routes = Routes + [route_ar]
        Policies = Policies + [policy]
        Costs = Costs + [cost]
        driving = driving + [driving_time]
        driving_only = driving_only + [driving_only_time]
        recourse = recourse + [policy_time]
    Duration['driving'] = driving  # driving includes driving + recourse
    Duration['driving_only'] = driving_only
    Duration['recourse'] = recourse

    # handling and waiting time are not dependent on the route
    for i in range(k):
        Duration.loc[i, 'handling'] = sum(2 * a * mu[Districts_0[i]])
        Duration.loc[i, 'waiting'] = sum(mu[Districts_0[i]] * a * (
                0.5 * p_w[Districts_0[i], 0] + 1.5 * p_w[Districts_0[i], 1] + 2.5 * p_w[
            Districts_0[i], 2]))

    if Summary == 'yes':
        print('Routes per district : ', Routes,
              '\nRecourse policies : ', Policies,
              '\nCosts : ', Costs,
              '\n\nDuration of route in hours : \n', Duration)

        TotalCosts = sum(Costs)

        print('\nTOTALS',
              '\nCosts : ', TotalCosts,
              '\nLongest duration : ', max(Duration.sum(axis=1)), ' hours')

    # Plot results #
    # create one array summarizing all routes where 'ar1' and 'ar2' are replaced with zeros
    Plot_list = np.array(Routes, dtype=object)
    for j in range(k):
        for i in range(len(Plot_list[j])):
            if Plot_list[j][i] in ['ar1', 'ar2']:
                Plot_list[j][i] = 0
    Plot_list = np.concatenate(Plot_list, axis=0)

    # create list with all the active arcs
    active_arcs = []
    for i in range(len(Plot_list) - 1):
        active_arcs.append((Plot_list[i], Plot_list[i + 1]))
    active_arcs.append(
        (Plot_list[-1],
         Plot_list[0]))  # add return to depot manually (only useful when preventive break in last route)

    # plot arcs
    plt.figure()
    for i, j in active_arcs:
        plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='black', zorder=0)
    plt.plot(xc[0], yc[0], c='black', marker='s')
    plt.scatter(xc[1:], yc[1:], c='black')
    plt.title('Routing, Iteratie ' + str(it))
    # plt.show()

    return Routes, Policies, Costs, Duration


# --------------------------- Iterative Procedure 2 ------------------------ #
# Decrease number of districts when it has been incorrectly initialized
def Check_number_districts(T, k, K, m_reason, Duration, b, mu, sd, k_tested):
    print('\n-----IP 2: CHECK NUMBER OF DISTRICTS-----')  # only checks for decreasing k
    decrease_k = 0
    Correct_number_m = 'yes'
    k_new = k
    K_new = K
    if m_reason != '(c)':
        k_time = (sum(Duration['driving']) + sum(b * mu) + st.norm.ppf(1 - Delta) * np.sqrt(
            sum((np.array(b) ** 2) * (sd ** 2)))) / T
        k_test = math.ceil(k_time)
        print('Approximation 2: ', str(k_test))
        print('k at previous iteration: ', str(k))
        if k_test != k:
            if k_test not in k_tested:
                if k_test < k and m_reason == '(t)':
                    Correct_number_m = 'no'
                    k_new = k_test
                    K_new = [i for i in range(1, k_new + 1)]
                    print('ITERATION: ADJUST NUMBER OF DISTRICTS because of inaccurate approximation at start')
                    print('Old number of districts: ', k)
                    print('New number of districts: ', k_new)
                    decrease_k = 1
    return Correct_number_m, k_new, K_new, decrease_k


# --------------------------- Iterative Procedure 3 ------------------------ #
# Exclusion of infeasible customer combinations whe the time chance constraint is violated
# if Equity == 'yes', we also include checking the feasibility of duration equality. (this were preliminary tests, and is not included in Reusken et al. (2023))
def Check_Feasibility(T, b, k, Districts_0, Districts, Duration, mu, sd, Eta, Equity):
    print('\n-----CHANCE CONSTRAINT ON TIME-----')
    failure_cc = '' # for chance constraint
    failure_ec = '' # for equality constraint

    # feasibility of chance constraint : using normal
    Z = [((T - Duration['driving'][i]) - sum(mu[Districts_0[i]] * np.array(b)[Districts_0[i]])) / np.sqrt(sum(
        (np.array(b)[Districts_0[i]] ** 2) * (sd[Districts_0[i]] ** 2))) for i in range(k)]  # derive z-score
    Chance_T = 1 - st.norm.cdf(Z)  # gives the probability that the total duration exceeds T
    print('The probability that the total duration exceeds', T, 'hours is', Chance_T * 100, '%')

    cc_failing = []  # set of districts for which the chance constraint is infeasible
    print('Chance Constraint:')
    for i in range(k):
        if Chance_T[i] <= Beta:
            print('District ', i, ': Satisfied')
        if Chance_T[i] >= Beta:
            print('District ', i, ': NOT Satisfied --> Duration of route is a problem')
            cc_failing = cc_failing + [Districts[i]]

    # feasibility of duration equity constraint
    ec_failing = []  # set of districts for which the equity constraint is infeasible
    if Equity == 'yes':
        z_test = [[(p, t), Duration.sum(axis=1)[p] - Duration.sum(axis=1)[t]] for p in range(k) for t in range(k) if
                  p > t]
        print('Duration equity:')
        for f in range(int((k * k - k) / 2)):
            if abs(z_test[f][1]) <= Eta:
                print('Districts ', z_test[f][0], ': Satisfied')
            if abs(z_test[f][1]) > Eta:
                print('Districts ', z_test[f][0], ': NOT Satisfied --> Equity across districts is a problem')
                if z_test[f][1] > 0:
                    ec_failing = ec_failing + [Districts[z_test[f][0][0]]]  # refers to first stored district
                if z_test[f][1] < 0:
                    ec_failing = ec_failing + [Districts[z_test[f][0][1]]]  # refers to second stored district

    # store which constraints have failed in this iteration:
    if cc_failing != []:
        failure_cc = 'y'
    if ec_failing != []:
        failure_ec = 'y'

    # fill K_adjust:
    K_adjust = cc_failing
    [K_adjust.append(x) for x in ec_failing if x not in K_adjust]  # no duplicates
    print(K_adjust)

    ## store reason --> currently not included, but if the equality is of interest, it might be useful to store the reason of failure
    # if ec_failing == []:
    #     if cc_failing != []:
    #         reason = 'chance constraint'
    # if cc_failing == []:
    #     if ec_failing != []:
    #         reason = 'equity constraint'
    # if cc_failing != []:
    #     if ec_failing != []:
    #         reason = 'both constraints'

    return K_adjust, failure_cc, failure_ec, Chance_T
