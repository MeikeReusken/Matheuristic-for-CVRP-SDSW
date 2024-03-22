from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from main import *


# --- REAL INSTANCES --- #

## save figures in one document
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


## Real-life data
df_raw = pd.read_excel('Data/Real-life instances/case_study_locations.xlsx')
df_collection_CAN = pd.read_excel('Data/Real-life instances/Historic_collection_amounts.xlsx',
                                  sheet_name='Moisson Montreal')
df_collection_NL = pd.read_excel('Data/Real-life instances/Historic_collection_amounts.xlsx', sheet_name='Eemsdelta')

print('Minimum collection amount: ', min(df_collection_CAN['collection']), min(df_collection_NL['collection']))
print('Maximum collection amount: ', max(df_collection_CAN['collection']), max(df_collection_NL['collection']))

sd_list = [0.0, 0.1, 0.2, 0.3]
rnd = np.random
for ie in sd_list:
    plt.rcParams["figure.figsize"] = [6.00, 6.00]
    plt.rcParams["figure.autolayout"] = True
    # Turn interactive plotting off
    plt.ioff()
    Real_instances = pd.DataFrame(index=range(9),
                                  columns=['|N|', '|K|', 'A3', 'assignment objective', 'A2', 'A1',
                                           'maxdur', 'mindur', 'meandur', 'time', 'time clustering', 'time routing',
                                           'assignment gap', 'iterations', 'solution found',
                                           'adjust num districts',
                                           'decrease num districts',
                                           'time chance constraint',
                                           'equity constraint', 'b', 'rho', 'End_reason',
                                           'driving', 'recourse', 'waiting', 'handling', 'Duration_total',
                                           'preventive recourse', 'risk recourse'])

    Uncertainty_table = pd.DataFrame(index=range(9),
                                     columns=['|N|', 'gamma-test', 'delta-test', 'alpha-test', 'alpha_av', 'eta-test', 'eta_av'])

    for d in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if d in [1, 2, 3, 4, 5, 6]:
            track_data_set = 'Moisson_Montreal'
            Q = 4535
            T = 8
            a = 0.0008
            df = pd.concat([df_raw.loc[df_raw['Instance'] == 0],
                           df_raw.loc[df_raw['Instance'] == d]])
            random.seed(d)
            mu = np.array(random.sample(range(round(min(df_collection_CAN['collection'])), round(
                max(df_collection_CAN['collection']))), len(df)-1))
        if d in [7, 8]:
            track_data_set = 'Eemsdelta'
            Q = 1000
            T = 4
            a = 0.008
            df = pd.concat([df_raw.loc[df_raw['Instance'] == 'a'],
                           df_raw.loc[df_raw['Instance'] == d]])
            random.seed(d)
            mu = np.array(random.sample(range(round(min(df_collection_NL['collection'])), round(
                max(df_collection_NL['collection']))), len(df)-1))
        if d in [9]:
            track_data_set = 'Zevenaar'
            Q = 2500
            T = 4
            a = 0.008
            df = pd.concat([df_raw.loc[df_raw['Instance'] == 'b'],
                           df_raw.loc[df_raw['Instance'] == d]])
            random.seed(d)
            mu = np.array(random.sample(range(round(min(df_collection_NL['collection'])), round(
                max(df_collection_NL['collection']))), len(df)-1))

        print('\n--- Data set: ', track_data_set)

        print('Daily time limit in hours: T = ', T)
        print('Service time units in hours per demand unit: a = ', a)
        print('Vehicle capacity: Q = ', Q)

        print(df)

        print('\nMean demands: ', mu)

        yc = np.array(df['latitude'])
        xc = np.array(df['longitude'])
        sd = ie * mu
        n = len(df) - 1

        output_table = pd.DataFrame()

        Findings, output, C_Duration, Uncertainty_res = Math(
            n, xc, yc, mu, sd, Q, 'convex', track_data_set, T, a)
        Real_instances.loc[d - 1] = Findings
        Uncertainty_table.loc[d - 1] = Uncertainty_res
    Real_instances['data set'] = ['Moisson Montreal', 'Moisson Montreal', 'Moisson Montreal',
                                  'Moisson Montreal', 'Moisson Montreal', 'Moisson Montreal', 'Eemsdelta', 'Eemsdelta', 'Zevenaar']
    Uncertainty_table['Data set'] = ['Moisson Montreal', 'Moisson Montreal', 'Moisson Montreal',
                                     'Moisson Montreal', 'Moisson Montreal', 'Moisson Montreal', 'Eemsdelta', 'Eemsdelta', 'Zevenaar']

    Real_instances = Real_instances[['data set', '|N|', '|K|', 'A1', 'A2', 'A3',
                                     'mindur', 'meandur', 'maxdur',
                                     'solution found', 'assignment gap',
                                     'iterations', 'adjust num districts', 'decrease num districts',
                                     'time chance constraint', 'driving', 'recourse', 'waiting', 'handling', 'Duration_total', 'preventive recourse', 'risk recourse', 'time', 'time clustering', 'time routing']]

    Real_instances.to_excel(f'Output/Main-real{ie}.xlsx')

    Uncertainty_table.to_excel(f'Output/Uncertainty-real{ie}.xlsx')

    ## save figures in one document
    save_multi_image(str(ie) + "Output/Plots_real.pdf")
    plt.close('all')
