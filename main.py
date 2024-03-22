import random
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import *

# variable inputs: xc, xy, n, mu, sd, Q, T, a

# Matheuristic
def Math(n, xc, yc, mu, sd, Q, constraint_type, track_data_set, T, a):
    Iterations = -1
    k_tested = []
    start_time = time.perf_counter()
    # get data
    p_w, N, V, c, b = Data(track_data_set, a, xc, yc, n, Plot='no')

    # subproblem 1: determine number of districts k
    k, K, reason_m, A1 = starting_k(track_data_set, xc, yc, Q, c, b,
                                    mu, sd, T, n,
                                    method='approx_TSP_Beardwood')  # method from {'solve_TSP', 'approx_TSP_Beardwood'}

    Correct_k = 'no'
    Feas_constraint_nd = 0
    Feas_constraint_nd_decrease = 0
    total_time_clustering = 0
    total_time_routing = 0
    while Correct_k == 'no':
        # find seeds
        k_tested = k_tested + [k]
        print('\nk_tested:')
        print(k_tested)
        # subproblem 2a: seed selection
        N_s, N_s_1, cone_allocation, xk, yk = Seeds(N, c, k, K,
                                                    xc, yc, n, track_data_set,
                                                    method='KMeans')  # method from {'Fisher','Sultana','KMeans'}

        Condition = 'not satisfied'
        Feasibility = 'no'
        K_violated = []
        x_previous = list(np.ones(n * k))
        Feas_constraint_cc = 0
        Feas_constraint_ec = 0
        E = b_equity  # hours allowed between the durations
        rho = 0
        rho_round = 0
        Ending_reason = ''
        Complete_Duration = pd.DataFrame(
            columns=['iteration', 'driving', 'handling', 'waiting', 'total expected duration', 'policy'])

        while Condition == 'not satisfied':
            Iterations = Iterations + 1
            print('\n------ Iteration : ', Iterations)
            start_time_clustering = time.perf_counter()
            # subproblem 2b: assignment of customers to districts
            Districts, Districts_0, Clustering_objective, Clustering_gap, Clustering_solution, Ass_status, C_all = Clustering(
                track_data_set,
                Q, N,
                c, b,
                k,
                K,
                N_s_1,
                xk,
                yk,
                xc,
                yc,
                n,
                mu,
                sd,
                T,
                Iterations,
                Capacity_type=constraint_type,
                # Capacity_type from {'linear', 'convex}
                Equity='no',
                Violated=K_violated,
                previous=x_previous,
                Eta=E,
                Reduce_variables='yes',
                m_reason=reason_m)
            end_time_clustering = time.perf_counter()
            total_time_clustering = total_time_clustering + \
                (end_time_clustering - start_time_clustering)
            # Iterative Procedure 1 (IP1): increase k when the assigment of customers to districts is infeasible
            if Ass_status == 'infeasible' or Ass_status == 'no_sol_in_time':
                # increasing k:
                print('Assignment problem failed - INCREASE K: ',
                      str(k), ' --> ', str(k + 1))
                k = k + 1
                K = [i for i in range(1, k + 1)]
                Feas_constraint_nd = Feas_constraint_nd + 1
                end_time = time.perf_counter()
                Findings = [n, k, "", "inf", "inf",
                            end_time - start_time,  # in seconds
                            "inf",  # in percentages
                            Iterations, "inf", "", ""]
                Condition = 'satisfied'
                Ending_reason = Ass_status

                end_time_stop = time.perf_counter()

                Findings = [n, str(k) + reason_m, 'no', 'no',
                            'no', A1,
                            'no', 'no', 'no',
                            end_time - start_time,  # in seconds
                            total_time_clustering,
                            total_time_routing,
                            'no',  # in percentages
                            Iterations, 'no', 'no', 'no', 'no', 'no', 'no', rho_round, Ending_reason]

            else:
                start_time_routing = time.perf_counter()
                # subproblem 3: routing
                Routes, Policies, Costs, Duration = Routing(track_data_set, a, p_w, k, Districts, Districts_0,
                                                            xc, yc, mu, sd, Q, Iterations, Summary='yes')
                end_time_routing = time.perf_counter()
                total_time_routing = total_time_routing + \
                    (end_time_routing - start_time_routing)

                temporary_df = pd.DataFrame(
                    columns=['iteration', 'driving', 'handling', 'waiting', 'total expected duration', 'policy'])
                temporary_df[['driving', 'handling', 'waiting',
                              'driving_only', 'recourse']] = Duration
                temporary_df['iteration'] = Iterations
                temporary_df['total expected duration'] = temporary_df['driving'] + temporary_df['handling'] + temporary_df[
                    'waiting']
                temporary_df['policy'] = Policies
                Complete_Duration = pd.concat(
                    [Complete_Duration, temporary_df])

                total_handling = temporary_df['handling'].sum()
                total_waiting = temporary_df['waiting'].sum()
                total_recourse = temporary_df['recourse'].sum()
                total_driving_only = temporary_df['driving_only'].sum()
                total_all = total_handling + total_waiting + total_recourse + total_driving_only
                print('Handling: ', total_handling)
                print('Waiting: ', total_waiting)
                print('Recourse: ', total_recourse)
                print('Driving only: ', total_driving_only)
                print('Total duration: ', total_all)

                # count policies
                unique_policy, counts_policy = np.unique(
                    Policies, return_counts=True)
                policy_dic = dict(zip(unique_policy, counts_policy))
                # total_preventive, total_risk = [], []
                if 'preventive break' in Policies:
                    total_preventive = policy_dic['preventive break']
                else:
                    total_preventive = 0
                if 'risk cost of a return trip' in Policies:
                    total_risk = policy_dic['risk cost of a return trip']
                else:
                    total_risk = 0

                # Iterative Procedure 2 (IP2): Decrease number of districts when it has been incorrectly initialized
                Correct_k, k, K, deck = Check_number_districts(
                    T, k, K, reason_m, Duration, b, mu, sd, k_tested)

                Feas_constraint_nd_decrease = Feas_constraint_nd_decrease + deck

                fail_cc = ''
                fail_ec = ''
                if Correct_k == 'yes':
                    # Iterative Procedure 3 (IP3): Exclusion of infeasible customer combinations when the time chance constraint is violated
                    K_violated_in_iteration, fail_cc, fail_ec, eta_check = Check_Feasibility(T, b, k, Districts_0, Districts, Duration,
                                                                                             mu, sd, Eta=E, Equity='no')

                time_check = time.perf_counter()
                running_time = time_check - start_time
                if running_time > 2 * 3600:  # limited running time of 2 hours
                    Condition = 'satisfied'

                else:
                    # count the number of iterations in which a certain constraint was infeasible
                    if Correct_k == 'no':
                        Feas_constraint_nd = Feas_constraint_nd + 1
                    if fail_cc == 'y':
                        Feas_constraint_cc = Feas_constraint_cc + 1
                    if fail_ec == 'y':
                        Feas_constraint_ec = Feas_constraint_ec + 1

                        # calculate rho : for duration equality
                        rho = Feas_constraint_ec / (2 * k)
                        rho_round = math.floor(rho)
                        if rho == rho_round:
                            # update b + theta* rho
                            E = E + theta * rho

                    if Correct_k == 'yes':
                        if K_violated_in_iteration == []:
                            Condition = 'satisfied'
                            Feasibility = 'yes'
                        if K_violated_in_iteration != []:
                            K_violated = K_violated + K_violated_in_iteration
                    if Iterations == 10 * k:
                        Condition = 'satisfied'

                end_time = time.perf_counter()
                # Store results
                Findings = [n, str(k) + reason_m,
                            sum(Costs) / s,  # A3
                            Clustering_objective,
                            (Clustering_objective + 2 * \
                             sum(c[0][N_s_1])) / s,  # A2
                            A1,
                            max(Duration.sum(axis=1)), min(Duration.sum(
                                axis=1)), sum(Duration.sum(axis=1)) / k,
                            end_time - start_time,  # in seconds
                            total_time_clustering,
                            total_time_routing,
                            Clustering_gap,  # in percentages
                            Iterations, Feasibility, Feas_constraint_nd, Feas_constraint_nd_decrease, Feas_constraint_cc, Feas_constraint_ec, E,
                            rho_round, Ending_reason, total_driving_only, total_recourse, total_waiting, total_handling, total_all, total_preventive, total_risk]

                if all(v == 0 for v in sd) == False:
                    gamma_test = 1 - \
                        st.norm.cdf((k*Q-sum(mu))/np.sqrt(sum(sd ** 2)))
                    delta_test = 1-st.norm.cdf((k*T - sum(Duration['driving'])-sum(
                        b * mu))/np.sqrt(sum((np.array(b) ** 2) * (sd ** 2))))
                    alpha_Z = [(Q - sum(mu[Districts_0[i]])) /
                               np.sqrt(sum(sd[Districts_0[i]] ** 2)) for i in range(k)]
                    alpha_test = 1 - st.norm.cdf(alpha_Z)
                    alpha_test_av = np.nanmean(alpha_test)
                    eta_test = eta_check
                    eta_test_av = np.nanmean(eta_test)
                    Uncertainty_results = np.array(
                        [n, gamma_test, delta_test, alpha_test, alpha_test_av, eta_test, eta_test_av], dtype=object)
                if all(v == 0 for v in sd) == True:
                    Uncertainty_results = np.zeros(7)

    return Findings, C_all, Complete_Duration, Uncertainty_results
