import matplotlib.pyplot as plt
import numpy as np

# Unified subplot
fig, axs = plt.subplots(1,5, figsize=(18,4), sharex=False, gridspec_kw={'width_ratios': [1.2, 1.2, 1.2, 1.0, 1.2]})

#plt.subplots_adjust(wspace=0.01)

# Data configuration for WindyGridworld
estimators_windy = ['CPDIS1', 'CPDIS2', 'CPDIS3', 'CPDIS4']
metrics_windy = ['Bias', 'Variance', 'MSE', 'ESS']
sample_sizes_windy = ['100', '300', '500', '1000', '1500', '2000', '5000']
line_styles_windy = {
    'CPDIS1': ('orange', '--'),
    'CPDIS2': ('blue', '--'),
    'CPDIS3': ('red', '--'),
    'CPDIS4': ('purple', '--'),
}
label_estimators_windy = {
    'CPDIS1': 'Intervention: $\pi(.|s)$',
    'CPDIS2': 'Intervention: $MLE(\pi(.|s))$',
    'CPDIS3': 'Intervention: Qual.',
    'CPDIS4': 'No Intervention',
}

handles = []

# Plot metrics for WindyGridworld
for i, metric in enumerate(metrics_windy):
    for estimator in estimators_windy:
        mean_values = [np.mean(WindyGridworld_Int_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_windy]
        variances = [np.var(WindyGridworld_Int_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_windy]
        std_devs = np.sqrt(variances)

        color, linestyle = line_styles_windy[estimator]
        line = axs[i].errorbar(sample_sizes_windy, mean_values, yerr=std_devs,
                        label=label_estimators_windy[estimator], color=color, linestyle=linestyle, capsize=5)
        if i==0:
            handles.append(line)
            axs[i].set_xlabel('Number of Samples',fontsize=18)

    axs[i].set_ylabel(metric, fontsize=18)
    axs[i].set_xticks(sample_sizes_windy)
    axs[i].set_xticklabels(sample_sizes_windy, rotation=45)
    axs[i].set_yscale('log')
    #if i == 0:
    #    axs[i].legend()

# Data configuration for MIMIC
estimators_mimic = ['CIS1', 'IntIS1', 'IntIS2', 'IntIS3']
sample_sizes_mimic = ['100', '200', '300',  '400', '500']
line_styles_mimic = {
    'CIS1': ('purple', '-', 's'),
    'IntIS1': ('orange', '-', 'd'),
    'IntIS2': ('blue', '-', '^'),
    'IntIS3': ('red', '-', 'v'),
}
legend_labels_mimic = {
    'CIS1': 'No Intervention',
    'IntIS1': 'Intervention: $\pi(.|s)$',
    'IntIS2': 'Intervention: $MLE(\pi(.|s))$',
    'IntIS3': 'Intervention: Qual.',
}

# Plot variance for MIMIC
already_plotted = set()
for estimator in estimators_mimic:
    if estimator not in already_plotted:
        mean_values = [np.mean(MIMIC_Int_Metrics[n_eps][estimator]['Variance']) for n_eps in sample_sizes_mimic]
        variances = [np.var(MIMIC_Int_Metrics[n_eps][estimator]['Variance']) for n_eps in sample_sizes_mimic]
        std_devs = np.sqrt(variances)

        color, linestyle, marker = line_styles_mimic[estimator]
        axs[4].errorbar(sample_sizes_mimic, mean_values, yerr=std_devs, label=legend_labels_mimic[estimator],
                        color=color, linestyle=linestyle, marker=marker, capsize=5, markersize=5)
        already_plotted.add(estimator)

axs[4].set_xlabel('Number of Samples',fontsize=18)
axs[4].set_ylabel('Variance', fontsize=18)
axs[4].set_xticks(sample_sizes_mimic)
axs[4].set_xticklabels(sample_sizes_mimic, rotation=45)
axs[4].set_yscale('log')
#axs[4].legend()

# Common titles
fig.text(0.5, 0.95, 'WindyGridworld', ha='center', va='center', fontsize=20)
fig.text(0.925, 0.95, 'MIMIC', ha='center', va='center', fontsize=20)

fig.legend(handles, [est.get_label() for est in handles], loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Interventions.png', dpi=400, bbox_inches='tight')
plt.show()
