from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from main import *


# --- RANDOM INSTANCES --- #

# save figures in one document
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


print('\n\nSOLOMON instances')
T = 8  # daily time limit in hours
a = 0.008  # service time units per demand unit
print('Time limit in hours: T= ', T)
print('Service time units per demand unit: a= ', a)
sd_list = [0.0, 0.1, 0.2, 0.3]
for ie in sd_list:
    print('\nUNCERTAINTY SCENARIO: ', ie)
    plt.rcParams["figure.figsize"] = [6.00, 6.00]
    plt.rcParams["figure.autolayout"] = True
    # Turn interactive plotting off
    plt.ioff()
    # loop over c101, c201 and r101
    instance_sets = ['c101', 'c201', 'r101']
    n_options = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    for d in instance_sets:
        print('\n\nData set: ', str(d))
        track_data_set = d
        data_raw = pd.read_fwf('Data/Random Instances/' + d + '.txt')
        data = data_raw.loc[5:len(data_raw)].astype(float)
        data.columns = data_raw.loc[4]

        xc_raw = np.array(data['XCOORD.'])
        yc_raw = np.array(data['YCOORD.'])
        mu_raw = np.array(data['DEMAND'])
        sd_raw = ie * np.array(data['DEMAND'])

        Q = float(data_raw.iloc[2][1])

        Random_instances = pd.DataFrame(index=range(len(n_options)),
                                        columns=['|N|', '|K|', 'A3', 'assignment objective', 'A2', 'A1',
                                                 'maxdur', 'mindur', 'meandur', 'time', 'time clustering', 'time routing',
                                                 'assignment gap', 'iterations', 'solution found',
                                                 'adjust num districts',
                                                 'decrease num districts',
                                                 'time chance constraint',
                                                 'equity constraint', 'b', 'rho', 'End_reason',
                                                 'driving', 'recourse', 'waiting', 'handling', 'Duration_total', 'preventive recourse', 'risk recourse'])

        Uncertainty_table = pd.DataFrame(index=range(len(n_options)),
                                         columns=['|N|', 'gamma-test', 'delta-test', 'alpha-test', 'alpha_av', 'eta-test', 'eta_av'])

        output_table = pd.DataFrame()

        for e in range(len(n_options)):
            print('\nSubset |N|= ', str(n_options[e]))
            n = n_options[e]
            xc = xc_raw[0:(n_options[e] + 1)]
            yc = yc_raw[0:(n_options[e] + 1)]
            mu = mu_raw[1:(n_options[e] + 1)]
            sd = sd_raw[1:(n_options[e] + 1)]

            Findings, output, C_Duration, Uncertainty_res = Math(
                n, xc, yc, mu, sd, Q, 'convex', track_data_set, T, a)
            Random_instances.loc[e] = Findings
            Uncertainty_table.loc[e] = Uncertainty_res

        Random_instances['instance'] = range(1, len(n_options) + 1)
        Random_instances['data set'] = d
        Uncertainty_table['Data set'] = d

        if d == 'c101':
            Results_c101 = Random_instances[
                ['data set', 'instance', '|N|', '|K|', 'assignment objective', 'A1', 'A2', 'A3',
                 'mindur', 'meandur', 'maxdur',
                 'solution found', 'assignment gap',
                 'iterations', 'adjust num districts', 'decrease num districts',
                 'time chance constraint', 'driving', 'recourse', 'waiting', 'handling', 'Duration_total',
                 'preventive recourse', 'risk recourse', 'time', 'time clustering', 'time routing',
                 'End_reason']]

            Results_uncertainty_c101 = Uncertainty_table[[
                'Data set', '|N|', 'gamma-test', 'delta-test', 'alpha-test', 'alpha_av', 'eta-test', 'eta_av']]

        if d == 'c201':
            Results_c201 = Random_instances[
                ['data set', 'instance', '|N|', '|K|', 'assignment objective', 'A1', 'A2', 'A3',
                 'mindur', 'meandur', 'maxdur',
                 'solution found', 'assignment gap',
                 'iterations', 'adjust num districts', 'decrease num districts',
                 'time chance constraint', 'driving', 'recourse', 'waiting', 'handling', 'Duration_total',
                 'preventive recourse', 'risk recourse', 'time', 'time clustering', 'time routing',
                 'End_reason']]
            Results_uncertainty_c201 = Uncertainty_table[[
                'Data set', '|N|', 'gamma-test', 'delta-test', 'alpha-test', 'alpha_av', 'eta-test', 'eta_av']]

        if d == 'r101':
            Results_r101 = Random_instances[
                ['data set', 'instance', '|N|', '|K|', 'assignment objective', 'A1', 'A2', 'A3',
                 'mindur', 'meandur', 'maxdur',
                 'solution found', 'assignment gap',
                 'iterations', 'adjust num districts', 'decrease num districts',
                 'time chance constraint', 'driving', 'recourse', 'waiting', 'handling', 'Duration_total',
                 'preventive recourse', 'risk recourse', 'time', 'time clustering', 'time routing',
                 'End_reason']]
            Results_uncertainty_r101 = Uncertainty_table[[
                'Data set', '|N|', 'gamma-test', 'delta-test', 'alpha-test', 'alpha_av', 'eta-test', 'eta_av']]

    Results_convex = pd.concat([Results_c101, Results_c201, Results_r101])
    Results_convex.to_excel(f'Output/Main-random{ie}.xlsx')
    # print(Results_convex)

    Results_uncertainty = pd.concat(
        [Results_uncertainty_c101, Results_uncertainty_c201, Results_uncertainty_r101])
    Results_uncertainty.to_excel(f'Output/Uncertainty-random{ie}.xlsx')

    # save figures in one document
    save_multi_image(f"Output/Plots_random{ie}.pdf")
    plt.close('all')
