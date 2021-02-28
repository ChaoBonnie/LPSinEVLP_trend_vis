import pandas as pd
import seaborn as sns; sns.set(style='white', context='paper')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import sem, t
from scipy import stats
from sklearn.metrics import roc_curve, auc
from numpy import median
from statannot import add_stat_annotation


### Create a spaghetti plot of two groups of lines
# according to whether LPS level at the second hour was less or greater than LPS level at the first hour ###


colors = sns.color_palette()[0:2]
df = pd.read_excel('C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/LPS 2hr Test/LPS Trend in EVLP.xlsx', sheet_name='Updated')
data = df[['LPS 1hr', 'LPS 2hr', 'LPS 4hr']].values

one_lt_two_label_set = False
one_gt_two_label_set = False
for onehr, twohr, fourhr in data:
    label = None
    if onehr < twohr:
        color = colors[0]
        if not one_lt_two_label_set:
            label = '1hr < 2hr'
            one_lt_two_label_set = True
    else:
        color = colors[1]
        if not one_gt_two_label_set:
            label = '1hr > 2hr'
            one_gt_two_label_set = True

    plt.plot(['1hr', '2hr', '4hr'], [onehr, twohr, fourhr], color=color, label=label, alpha=0.5)

plt.xlabel('EVLP Time')
plt.ylabel('LPS Level (EU/mL)')
plt.legend()
plt.show()

plt.savefig('Updated Spaghetti Plot.png', dpi=200)
plt.close()


### Acquire individual best-fit slope from LPS trend in each EVLP case ###


df_LPS = pd.read_excel(r'C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/LPS 2hr Test/Updated LPS Data.xlsx', sheet_name='Corrected for Plate 1 Standards')
df_LPS = df_LPS.dropna(subset=['EVLP 1hr', 'EVLP 2hr'])
df_LPS = df_LPS.reset_index(drop=True)
print(df_LPS)

df_LPStrend = pd.read_excel(r'C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/LPS 2hr Test/Updated LPS Data.xlsx',
                            sheet_name='Calculate Trend')
df_LPStrend_dec = df_LPStrend[df_LPStrend['Outcome'] == 0]
df_LPStrend_tx = df_LPStrend[df_LPStrend['Outcome'] == 1]
print(df_LPStrend_dec, df_LPStrend_tx)

slopes_dec, slopes_tx, slopes_all = [], [], []
for case_dec in df_LPStrend_dec['EVLP ID']:
    m_dec, _, _, _, _ = stats.linregress(df_LPStrend_dec['EVLP Time Point (hr)'][df_LPStrend_dec['EVLP ID'] == case_dec],
                                         df_LPStrend_dec['LPS Level (EU/mL)'][df_LPStrend_dec['EVLP ID'] == case_dec])
    slopes_dec.append(m_dec)
for case_tx in df_LPStrend_tx['EVLP ID']:
    m_tx, _, _, _, _ = stats.linregress(df_LPStrend_tx['EVLP Time Point (hr)'][df_LPStrend_tx['EVLP ID'] == case_tx],
                                        df_LPStrend_tx['LPS Level (EU/mL)'][df_LPStrend_tx['EVLP ID'] == case_tx])
    slopes_tx.append(m_tx)
for case in df_LPStrend['EVLP ID']:
    m_all, _, _, _, _ = stats.linregress(df_LPStrend['EVLP Time Point (hr)'][df_LPStrend['EVLP ID'] == case],
                                        df_LPStrend['LPS Level (EU/mL)'][df_LPStrend['EVLP ID'] == case])
    slopes_all.append(m_all)
slopes_all_no_repeats = slopes_all[::3]
df_slopes_all_no_repeats = pd.DataFrame(slopes_all_no_repeats)
df_slopes_all_no_repeats.to_csv('EVLP-LPS Slopes.csv')
print(len(slopes_dec), '\n', len(slopes_tx))
print('Mean of best-fit slope (Dec): ', np.mean(slopes_dec), '\n Mean of best-fit slope (Tx): ', np.nanmean(slopes_tx))


### Create a spaghetti plot of LPS levels at the first, second, and fourth hour of EVLP ###


df_spaghetti = pd.read_excel(r'C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/LPS 2hr Test/Updated LPS Data.xlsx',
                             sheet_name='Calculate Trend')

# Spaghetti plot for all cases #

fig_spaghetti = plt.figure(figsize=(6, 3))
for case in df_spaghetti.index:
    if df_spaghetti.loc[case, 'Outcome'].all() == 1:
        color = 'skyblue'
    elif df_spaghetti.loc[case, 'Outcome'].all() == 0:
        color = 'lightsalmon'
    plt.plot(df_spaghetti.loc[case, 'EVLP Time Point (hr)'],
         df_spaghetti.loc[case, 'LPS Level (EU/mL)'], color=color, alpha=0.4)
plt.title('LPS Trend Over EVLP (Tx + Dec)')
plt.xlabel('EVLP Time Point (hr)')
plt.ylabel('LPS Level (EU/mL)')
# plt.legend(['Transplanted', 'Declined'])
fig_spaghetti.tight_layout()
fig_spaghetti.savefig('Spaghetti Plot (Dec + Tx)' + '.jpg', dpi=200)
plt.close()

# Spaghetti plot for declined cases #

fig_spaghetti_dec = plt.figure(figsize=(6, 3))
for case in df_spaghetti['EVLP ID'][df_spaghetti['Outcome'] == 0]:
    sns.lineplot(df_spaghetti['EVLP Time Point (hr)'][df_spaghetti['EVLP ID'] == case],
         df_spaghetti['LPS Level (EU/mL)'][df_spaghetti['EVLP ID'] == case], color='lightsalmon', alpha=0.15, zorder=1)
g = sns.lineplot(data=df_spaghetti, x=df_spaghetti['EVLP Time Point (hr)'][df_spaghetti['Outcome'] == 0],
                 y=df_spaghetti['LPS Level (EU/mL)'][df_spaghetti['Outcome'] == 0], estimator=median, color='black', zorder=1000,
                 err_kws={'linestyle': '--', 'edgecolor': 'black', 'linewidth': 0.75, 'facecolor': 'none', 'alpha': 1,
                          'antialiased': True, 'zorder': 1001})
plt.title('LPS Trend Over EVLP (Dec)')
plt.xlabel('EVLP Time Point (hr)')
plt.ylabel('LPS Level (EU/mL)')
plt.xlim(left=1, right=4)
fig_spaghetti_dec.tight_layout()
fig_spaghetti_dec.savefig('Spaghetti Plot (Dec)' + '.jpg', dpi=200)
plt.close()

# Spaghetti plot for transplanted cases #

fig_spaghetti_tx = plt.figure(figsize=(6, 3))
for case in df_spaghetti['EVLP ID'][df_spaghetti['Outcome'] == 1]:
    sns.lineplot(df_spaghetti['EVLP Time Point (hr)'][df_spaghetti['EVLP ID'] == case],
         df_spaghetti['LPS Level (EU/mL)'][df_spaghetti['EVLP ID'] == case], color='skyblue', alpha=0.1, zorder=1)
g = sns.lineplot(data=df_spaghetti, x=df_spaghetti['EVLP Time Point (hr)'][df_spaghetti['Outcome'] == 1],
                 y=df_spaghetti['LPS Level (EU/mL)'][df_spaghetti['Outcome'] == 1], estimator=median, color='black', zorder=1000,
                 err_kws={'linestyle': '--', 'edgecolor': 'black', 'linewidth': 0.75, 'facecolor': 'none', 'alpha': 1, 'antialiased': True,
                          'zorder': 1001})
plt.title('LPS Trend Over EVLP (Tx)')
plt.xlabel('EVLP Time Point (hr)')
plt.ylabel('LPS Level (EU/mL)')
plt.xlim(left=1, right=4)
fig_spaghetti_tx.tight_layout()
fig_spaghetti_tx.savefig('Spaghetti Plot (Tx)' + '.jpg', dpi=200)
plt.close()


### Plot Time Trend Boxplots ###


fig = plt.figure(figsize=(4, 6))
sns.boxplot(data=df_LPS.iloc[:,1:4], color="skyblue")
sns.pointplot(data=df_LPS.iloc[:,1:4][df_LPS['Outcome'] == 0], color='red', estimator=median, capsize=.1)
sns.pointplot(data=df_LPS.iloc[:,1:4][df_LPS['Outcome'] == 1], color='green', estimator=median, capsize=.1)
plt.ylabel('LPS Level (EU/mL)')
leg = plt.legend(labels=['Declined', 'Transplanted'])
leg.legendHandles[0].set_color('red')
leg.legendHandles[1].set_color('green')
fig.savefig('Updated LPS Level Boxplot' + '.jpg')
plt.close()


### Plot Trendline ###


fig = plt.figure(figsize=(4, 6))
sns.pointplot(data=df_LPS.iloc[:,1:4][df_LPS['Outcome'] == 0], color='red', estimator=median, capsize=.1)
sns.pointplot(data=df_LPS.iloc[:,1:4][df_LPS['Outcome'] == 1], color='green', estimator=median, capsize=.1)
# plt.xlabel('EVLP Hour')
plt.ylabel('LPS Level (EU/mL)')
leg = plt.legend(labels=['Declined', 'Transplanted'])
leg.legendHandles[0].set_color('red')
leg.legendHandles[1].set_color('green')
fig.savefig('LPS Trendline', + '.jpg')
plt.close()


### Plot Transplanted vs. Declined Boxplots ###


df_LPS['Outcome'] = df_LPS['Outcome'].replace({0:'Declined', 1:'Transplanted', 2: np.NaN})
fig_TxDec = plt.figure(figsize=(3, 6))
plt.ylabel('LPS Level in EVLP-4hr Perfusates (EU/mL)')
ax = sns.boxplot(data=df_LPS, x= 'Outcome', y='EVLP 4hr', palette='pastel')
sns.stripplot(data=df_LPS, x= 'Outcome', y='EVLP 4hr', alpha=0.7, ax=ax)
ax.set(ylabel='LPS Level in EVLP-4hr Perfusates (EU/mL)')
add_stat_annotation(ax=ax, data=df_LPS, x='Outcome', y='EVLP 4hr', width=0.4, box_pairs=[['Transplanted', 'Declined']],
                                perform_stat_test=True, test='Mann-Whitney',
                                loc='inside', verbose=0, no_ns=True, fontsize='large')
fig_TxDec.tight_layout()
fig_TxDec.savefig('LPS Tx-Dec Boxplot' + '.jpg')
plt.close()


### Calculate LPS changes over EVLP ###


df_LPS['LPS Change'] = (df_LPS['LPS 4hr'] - df_LPS['LPS 1hr']) / df_LPS['LPS 1hr']
print(df_LPS)

h = sem(df_LPS['LPS Change']) * t.ppf((1 + 0.95) / 2, len(df_LPS['LPS Change']) - 1)
mean = np.mean(df_LPS['LPS Change'])
print('Mean change in LPS: +{}% [95% CI: +{}% to +{}%]'.format(mean*100, (mean-h)*100, (mean+h)*100))

# Histogram of changes in LPS level #

sns.distplot(df_LPS['LPS Change'])
plt.xlabel('Change in LPS Level (EVLP 4hr - EVLP 1hr)')
plt.savefig('Change in LPS Level Distribution' + '.jpg')


### Plot ICU-LOS Boxplots ###


df_LPS = df_LPS[df_LPS['Outcome'] == 1]
df_LPS['LPS in EVLP-4hr Perfusates'] = pd.cut(df_LPS['EVLP 4hr'], bins=[-1, 0.17, 10], include_lowest=True, labels=['≤0.17 EU/mL', '>0.17 EU/mL'])
fig_ICU = plt.figure(figsize=(3, 6))
# plt.ylabel('ICU Length of Stay (Days)')
ax = sns.boxplot(data=df_LPS, x= 'LPS in EVLP-4hr Perfusates', y='ICU LOS', showfliers=False, palette='pastel')
ax = sns.stripplot(data=df_LPS, x= 'LPS in EVLP-4hr Perfusates', y='ICU LOS', alpha=0.7)
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 5
ax.set(ylabel='ICU Length of Stay (Days)')
fig_ICU.tight_layout()
fig_ICU.savefig('LPS ICU-LOS Boxplot' + '.jpg')
plt.close()


### Calculate mean, median, standard deviation, quartiles ###

dec_1hr = df_LPS.iloc[:,1][df_LPS['Outcome'] == 0]
dec_2hr = df_LPS.iloc[:,2][df_LPS['Outcome'] == 0]
dec_4hr = df_LPS.iloc[:,3][df_LPS['Outcome'] == 0]
tx_1hr = df_LPS.iloc[:,1][df_LPS['Outcome'] == 1]
tx_2hr = df_LPS.iloc[:,2][df_LPS['Outcome'] == 1]
tx_4hr = df_LPS.iloc[:,3][df_LPS['Outcome'] == 1]
print('Mean of EVLP-1hr LPS Levels (Declined Cases): ', np.mean(dec_1hr))
print('Mean of EVLP-4hr LPS Levels (Declined Cases): ', np.mean(dec_4hr))
print('Mean of EVLP-1hr LPS Levels (Transplanted Cases): ', np.mean(tx_1hr))
print('Mean of EVLP-4hr LPS Levels (Transplanted Cases): ', np.mean(tx_4hr))
print('Median of EVLP-1hr LPS Levels (Declined Cases): ', np.median(dec_1hr))
print('Median of EVLP-2hr LPS Levels (Declined Cases): ', np.median(dec_2hr))
print('Median of EVLP-4hr LPS Levels (Declined Cases): ', np.median(dec_4hr))
print('Median of EVLP-1hr LPS Levels (Transplanted Cases): ', np.median(tx_1hr))
print('Median of EVLP-2hr LPS Levels (Transplanted Cases): ', np.median(tx_2hr))
print('Median of EVLP-4hr LPS Levels (Transplanted Cases): ', np.median(tx_4hr))

print('STD of EVLP-1hr LPS Levels (Declined Cases): ', np.std(dec_1hr))
print('STD of EVLP-4hr LPS Levels (Declined Cases): ', np.std(dec_4hr))
print('STD of EVLP-1hr LPS Levels (Transplanted Cases): ', np.std(tx_1hr))
print('STD of EVLP-4hr LPS Levels (Transplanted Cases): ', np.std(tx_4hr))
print('Q1 and Q3 of EVLP-1hr LPS Levels (Declined Cases): ', np.quantile(dec_1hr, 0.25), np.quantile(dec_1hr, 0.75))
print('Q1 and Q3 of EVLP-4hr LPS Levels (Declined Cases): ', np.quantile(dec_4hr, 0.25), np.quantile(dec_4hr, 0.75))
print('Q1 and Q3 of EVLP-1hr LPS Levels (Declined Cases): ', np.quantile(dec_2hr, 0.25), np.quantile(dec_2hr, 0.75))
print('Q1 and Q3 of EVLP-4hr LPS Levels (Declined Cases): ', np.quantile(tx_2hr, 0.25), np.quantile(tx_2hr, 0.75))
print('Q1 and Q3 of EVLP-1hr LPS Levels (Transplanted Cases): ', np.quantile(tx_1hr, 0.25), np.quantile(tx_1hr, 0.75))
print('Q1 and Q3 of EVLP-4hr LPS Levels (Transplanted Cases): ', np.quantile(tx_4hr, 0.25), np.quantile(tx_4hr, 0.75))


### Significance and equal-variance tests ###


print('2hr (Tx) vs. 2hr (Dec)', stats.mannwhitneyu(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 2hr'][df_LPS['Outcome']==1]))
print('1hr (Tx) vs. 1hr (Dec)', stats.mannwhitneyu(df_LPS['EVLP 1hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==1]))
print('4hr (Tx) vs. 4hr (Dec)', stats.mannwhitneyu(df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))
print('2hr (Dec) vs. 1hr (Dec)', stats.mannwhitneyu(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==0]))
print('2hr (Dec) vs. 4hr (Dec)', stats.mannwhitneyu(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 4hr'][df_LPS['Outcome']==0]))
print('4hr (Dec) vs. 1hr (Dec)', stats.mannwhitneyu(df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==0]))
print('2hr (Tx) vs. 1hr (Tx)', stats.mannwhitneyu(df_LPS['EVLP 2hr'][df_LPS['Outcome']==1], df_LPS['EVLP 1hr'][df_LPS['Outcome']==1]))
print('2hr (Tx) vs. 4hr (Tx)', stats.mannwhitneyu(df_LPS['EVLP 2hr'][df_LPS['Outcome']==1], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))
print('1hr (Tx) vs. 4hr (Tx)', stats.mannwhitneyu(df_LPS['EVLP 1hr'][df_LPS['Outcome']==1], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))

print('\n 2hr (Tx) vs. 2hr (Dec)', stats.levene(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 2hr'][df_LPS['Outcome']==1]))
print('1hr (Tx) vs. 1hr (Dec)', stats.levene(df_LPS['EVLP 1hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==1]))
print('4hr (Tx) vs. 4hr (Dec)', stats.levene(df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))
print('2hr (Dec) vs. 1hr (Dec)', stats.levene(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==0]))
print('2hr (Dec) vs. 4hr (Dec)', stats.levene(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 4hr'][df_LPS['Outcome']==0]))
print('4hr (Dec) vs. 1hr (Dec)', stats.levene(df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==0]))
print('2hr (Tx) vs. 1hr (Tx)', stats.levene(df_LPS['EVLP 2hr'][df_LPS['Outcome']==1], df_LPS['EVLP 1hr'][df_LPS['Outcome']==1]))
print('2hr (Tx) vs. 4hr (Tx)', stats.levene(df_LPS['EVLP 2hr'][df_LPS['Outcome']==1], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))
print('1hr (Tx) vs. 4hr (Tx)', stats.levene(df_LPS['EVLP 1hr'][df_LPS['Outcome']==1], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))

print('\n 2hr (Tx) vs. 2hr (Dec)', stats.ttest_ind(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 2hr'][df_LPS['Outcome']==1]))
print('1hr (Tx) vs. 1hr (Dec)', stats.ttest_ind(df_LPS['EVLP 1hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==1]))
print('4hr (Tx) vs. 4hr (Dec)', stats.ttest_ind(df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1], equal_var=False))
print('2hr (Dec) vs. 1hr (Dec)', stats.ttest_ind(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==0], equal_var=False))
print('2hr (Dec) vs. 4hr (Dec)', stats.ttest_ind(df_LPS['EVLP 2hr'][df_LPS['Outcome']==0], df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], equal_var=False))
print('4hr (Dec) vs. 1hr (Dec)', stats.ttest_ind(df_LPS['EVLP 4hr'][df_LPS['Outcome']==0], df_LPS['EVLP 1hr'][df_LPS['Outcome']==0], equal_var=False))
print('2hr (Tx) vs. 1hr (Tx)', stats.ttest_ind(df_LPS['EVLP 2hr'][df_LPS['Outcome']==1], df_LPS['EVLP 1hr'][df_LPS['Outcome']==1]))
print('2hr (Tx) vs. 4hr (Tx)', stats.ttest_ind(df_LPS['EVLP 2hr'][df_LPS['Outcome']==1], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))
print('1hr (Tx) vs. 4hr (Tx)', stats.ttest_ind(df_LPS['EVLP 1hr'][df_LPS['Outcome']==1], df_LPS['EVLP 4hr'][df_LPS['Outcome']==1]))