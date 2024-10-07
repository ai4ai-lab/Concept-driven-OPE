import matplotlib.pyplot as plt
import numpy as np

# Create a single figure for all subplots
fig, axs = plt.subplots(1,5, figsize=(18,4), sharex=False, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]})

# First four subplots using WindyGridWorld_Metrics data
estimators_1 = ['IS', 'PDIS', 'CIS', 'CPDIS']
metrics_1 = ['Bias', 'Variance', 'MSE', 'ESS']
sample_sizes_1 = ['100', '300', '500', '1000', '1500', '2000', '5000']
line_styles_1 = {
    'IS': ('black', '-'),
    'CIS': ('purple', '-'),
    'PDIS': ('black', '--'),
    'CPDIS': ('purple', '--'),
    'CIS1': ('purple', '-'),
    'CPDIS1': ('purple', '--'),
}
legend_labels = {
    'IS': 'IS',
    'PDIS': 'PDIS',
    'CIS': 'Parameterized CIS',
    'CPDIS': 'Parameterized CPDIS',
    'CIS1': 'Parameterized CIS',
    'CPDIS1': 'Parameterized CPDIS',
}

handles = []
labels = []

# Plot each metric for the first 4 subplots
for i, metric in enumerate(metrics_1):
    for estimator in estimators_1:
        mean_values = [np.mean(Unknown_Windygridworld_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_1]
        variances = [np.var(Unknown_Windygridworld_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_1]
        std_devs = np.sqrt(variances)

        color, linestyle = line_styles_1[estimator]
        line = axs[i].errorbar(sample_sizes_1, mean_values, yerr=std_devs, label=legend_labels[estimator], color=color, linestyle=linestyle, capsize=5)
        if i==0:
            handles.append(line)
            labels.append(line.get_label())
            axs[i].set_xlabel('Number of Samples', fontsize=18)

    axs[i].set_ylabel(metric, fontsize=18)  # Set as y-label
    #axs[i].set_xlabel('Number of Samples')
    axs[i].set_xticklabels(sample_sizes_1, rotation=45)
    axs[i].set_yscale('log')
    #if i == 0:
    #    axs[i].legend()

estimators_1 = ['CIS', 'CPDIS']
metrics_1 = ['Bias', 'Variance', 'MSE', 'ESS']
sample_sizes_1 = ['100', '300', '500', '1000', '1500', '2000', '5000']
line_styles_1 = {
    'CIS': ('green', '-'),
    'CPDIS': ('green', ':'),
}

legend_labels_1 = {
    'CIS': 'Known CIS',
    'CPDIS': 'Known CPDIS',
}

# Plot each metric for the first 4 subplots
for i, metric in enumerate(metrics_1):
    for estimator in estimators_1:
        mean_values = [np.mean(Windy_Gridworld_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_1]
        variances = [np.var(Windy_Gridworld_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_1]
        std_devs = np.sqrt(variances)

        color, linestyle = line_styles_1[estimator]
        line = axs[i].errorbar(sample_sizes_1, mean_values, yerr=std_devs, label=legend_labels_1[estimator], color=color, linestyle=linestyle, capsize=5)

        if i==0:
            handles.append(line)
            labels.append(line.get_label())

# Adding subtitles for groups
fig.text(0.5, 0.95, 'WindyGridWorld', ha='center', fontsize=20)
fig.text(0.925, 0.95, 'MIMIC', ha='center', fontsize=20)

# Fifth subplot using MIMIC_Metrics data
estimators_2 = ['IS', 'PDIS', 'CIS1', 'CPDIS1']
metrics_2 = ['Variance']
sample_sizes_2 = ['100', '200', '300', '400', '500']
line_styles_2 = {
    'IS': ('black', '-'),
    'PDIS': ('black', '--'),
    'CIS1': ('purple', '-'),
    'CPDIS1': ('purple', '--')
}
# Plot the metric for the fifth subplot
for metric in metrics_2:
    for estimator in estimators_2:
        mean_values = [np.mean(Unknown_MIMIC_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_2]
        variances = [np.var(Unknown_MIMIC_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_2]
        std_devs = np.sqrt(variances)

        color, linestyle = line_styles_2[estimator]
        axs[4].errorbar(sample_sizes_2, mean_values, yerr=std_devs, label=legend_labels[estimator], color=color, linestyle=linestyle, capsize=5)

    axs[4].set_xlabel('Number of Samples', fontsize=18)
    axs[4].set_ylabel('Variance', fontsize=18)
    axs[4].set_xticklabels(sample_sizes_2, rotation=45)
    axs[4].set_yscale('log')
    #axs[4].legend()

for metric in metrics_2:
    for estimator in estimators_2:
        mean_values = [np.mean(MIMIC_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_2]
        variances = [np.var(MIMIC_Metrics[n_eps][estimator][metric]) for n_eps in sample_sizes_2]
        std_devs = np.sqrt(variances)

        color, linestyle = line_styles_2[estimator]
        axs[4].errorbar(sample_sizes_2, mean_values, yerr=std_devs, label=legend_labels[estimator], color=color, linestyle=linestyle, capsize=5)

fig.legend(handles, [est.get_label() for est in handles], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for subtitles
plt.savefig('UnknownConcepts.png', dpi=400, bbox_inches='tight')
plt.show()
