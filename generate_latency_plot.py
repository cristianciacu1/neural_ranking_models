import matplotlib.pyplot as plt
import matplotlib

# Data for the plot
latency_fiqa = [13.46 + 15.09, 13.94 + 16.82, 13.1 + 16.51, 4.61+15.19, 73.46 + 17.13, 48.77 + 15.9]
recall_fiqa = [0.774, 0.769, 0.773, 0.748, 0.841, 0.733]

latency_nfcorpus = [3.9 + 9.16, 3.87 + 10.22, 3.72 + 9.57, 13.7 + 9.23, 24.18 + 23.59, 47.37 + 12.38]
recall_nfcorpus = [0.36, 0.363, 0.351, 0.352, 0.58, 0.445]

latency_scifact = [10.63 + 13.77, 10.57 + 13.53, 10.33 + 14.0, 3.9 + 13.63, 67, 46 + 14.1]
recall_scifact = [0.97, 0.97, 0.97, 0.96, 0.99, 0.97]

labels = ["BM25", "TF-IDF", "DeepCT", "DeepImpact", "SPLADE", "uniCOIL"]

# Markers and colors
markers = ["s", "o", "d", "^", "P", "*", "D"]
colors = ["gold", "purple", "lightgreen", "blue", "red", "brown"]

latency = latency_scifact
recall = recall_scifact

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 19}

matplotlib.rc('font', **font)

# FiQA
# plt.figure(figsize=(10, 6))
# for i in range(len(latency)):
#     if labels[i] not in ["DeepImpact", "SPLADE", "uniCOIL"]:  # Exclude DeepImpact and SPLADE from the legend
#         plt.scatter(latency[i], recall[i], label=labels[i], marker=markers[i], color=colors[i], s=200)  # Increase point size
#     else:
#         plt.scatter(latency[i], recall[i], marker=markers[i], color=colors[i], s=200)
#         if labels[i] == 'SPLADE':
#             plt.text(latency[i]-12, recall[i]-0.015, labels[i], fontsize=15)
#         elif labels[i] == 'DeepImpact':
#             plt.text(latency[i]+2, recall[i]-0.003, labels[i], fontsize=15)
#         elif labels[i] == 'uniCOIL':
#             plt.text(latency[i]-17, recall[i]-0.002, labels[i], fontsize=15)

# NFCorpus
# plt.figure(figsize=(10, 6))
# for i in range(len(latency)):
#     if labels[i] not in ["DeepImpact", "SPLADE", "uniCOIL"]:  # Exclude DeepImpact and SPLADE from the legend
#         plt.scatter(latency[i], recall[i], label=labels[i], marker=markers[i], color=colors[i], s=200)  # Increase point size
#     else:
#         plt.scatter(latency[i], recall[i], marker=markers[i], color=colors[i], s=200)
#         if labels[i] == 'SPLADE':
#             plt.text(latency[i]-3, recall[i]-0.025, labels[i], fontsize=15)
#         elif labels[i] == 'DeepImpact':
#             plt.text(latency[i]+1.5, recall[i]-0.004, labels[i], fontsize=15)
#         elif labels[i] == 'uniCOIL':
#             plt.text(latency[i]-8, recall[i]+0.009, labels[i], fontsize=15)

# SciFact
plt.figure(figsize=(10, 6))
for i in range(len(latency)):
    if labels[i] not in ["DeepImpact", "SPLADE", "uniCOIL"]:  # Exclude DeepImpact and SPLADE from the legend
        plt.scatter(latency[i], recall[i], label=labels[i], marker=markers[i], color=colors[i], s=200)  # Increase point size
    else:
        plt.scatter(latency[i], recall[i], marker=markers[i], color=colors[i], s=200)
        if labels[i] == 'SPLADE':
            plt.text(latency[i]-9, recall[i]-0.003, labels[i], fontsize=15)
        elif labels[i] == 'DeepImpact':
            plt.text(latency[i]+1.2, recall[i]-0.0004, labels[i], fontsize=15)
        elif labels[i] == 'uniCOIL':
            plt.text(latency[i]-4, recall[i]+0.0015, labels[i], fontsize=15)

# Adding labels and title with increased font size
plt.xlabel("Latency (ms/query)", fontweight='bold', fontsize=19)
plt.ylabel("R@1000", fontweight='bold', fontsize=19)

# Adding legend
plt.legend(loc='lower right', fontsize=14)

plt.subplots_adjust(left=0.425, bottom=0.304)

# Show plot with dotted grid lines
plt.grid(True, linestyle=':')  # Dotted grid lines
plt.show()