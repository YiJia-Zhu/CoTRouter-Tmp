import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
import matplotlib.ticker as mtick
import pandas as pd
import pickle
import itertools
import io

# --- Configuration (Keeping original) ---
config = {
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}

matplotlib.rc('pdf', fonttype=42)
FONTSIZE = 20
ALLWIDTH = 1.5
Marker = ['o', 'v', '8', 's', 'p', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X']
HATCH = ['+', 'x', '/', 'o', '|', '\\', '-', 'O', '.', '*']
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)


# --- Data Preparation ---
# Converting the single-line CSV data into a pandas DataFrame
csv_data = """Method,Dataset,Total Samples,Total Tokens,Total Switches,Avg Switches per Sample,Switches per 100 Tokens,Freq_01-05%,Freq_06-10%,Freq_11-15%,Freq_16-20%,Freq_21-25%,Freq_26-30%,Freq_31-35%,Freq_36-40%,Freq_41-45%,Freq_46-50%,Freq_51-55%,Freq_56-60%,Freq_61-65%,Freq_66-70%,Freq_71-75%,Freq_76-80%,Freq_81-85%,Freq_86-90%,Freq_91-95%,Freq_96-100%
Sweep_1,GSM8K,1319,1882601,47620,36.1031,2.5295,4.49%,4.02%,4.88%,5.17%,5.33%,5.32%,5.36%,5.49%,5.42%,5.51%,5.55%,5.58%,5.46%,5.43%,5.27%,4.97%,4.67%,4.25%,3.81%,4.01%
"""

# csv_data = """Method,Dataset,Total Samples,Total Tokens,Total Switches,Avg Switches per Sample,Switches per 100 Tokens,Freq_01-05%,Freq_06-10%,Freq_11-15%,Freq_16-20%,Freq_21-25%,Freq_26-30%,Freq_31-35%,Freq_36-40%,Freq_41-45%,Freq_46-50%,Freq_51-55%,Freq_56-60%,Freq_61-65%,Freq_66-70%,Freq_71-75%,Freq_76-80%,Freq_81-85%,Freq_86-90%,Freq_91-95%,Freq_96-100%
# Sweep_1,MATH,500,1691380,35766,71.5320,2.1146,5.41%,5.64%,5.59%,5.43%,5.39%,5.23%,5.28%,5.27%,5.09%,5.13%,5.02%,5.19%,5.05%,5.07%,4.99%,4.85%,4.58%,4.14%,3.84%,3.79%
# """


# csv_data = """Method,Dataset,Total Samples,Total Tokens,Total Switches,Avg Switches per Sample,Switches per 100 Tokens,Freq_01-05%,Freq_06-10%,Freq_11-15%,Freq_16-20%,Freq_21-25%,Freq_26-30%,Freq_31-35%,Freq_36-40%,Freq_41-45%,Freq_46-50%,Freq_51-55%,Freq_56-60%,Freq_61-65%,Freq_66-70%,Freq_71-75%,Freq_76-80%,Freq_81-85%,Freq_86-90%,Freq_91-95%,Freq_96-100%
# Sweep_1,AIME,30,202361,4955,165.1667,2.4486,5.15%,5.69%,5.49%,5.49%,5.63%,5.15%,5.13%,5.03%,5.19%,4.90%,4.72%,4.68%,4.84%,5.03%,5.21%,4.82%,4.54%,4.62%,4.26%,4.44%
# """
df = pd.read_csv(io.StringIO(csv_data))

# Extracting the fine-grained frequency data columns
freq_columns = df.columns[7:] 
frequencies_raw = df.iloc[0][freq_columns]

# Data Cleaning: Remove '%' and convert to float
frequencies = frequencies_raw.str.replace('%', '').astype(float).values

# Constructing X-axis labels (5% intervals)
# MODIFICATION 1: Simplify x_labels to '5%', '10%', etc.
new_x_labels = [f'{i*5}' for i in range(1, len(frequencies) + 1)]

x_positions = np.arange(len(frequencies))


# --- Create and Plot Chart (Modified Labels) ---
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the bar chart
bars = ax.bar(x_positions, frequencies, width=0.7, color='white',
              ec=COLORS[5], hatch=HATCH[2] * 2, linewidth=ALLWIDTH)

# --- Set Y-axis Limits (Keeping original logic) ---
max_freq = frequencies.max()
y_limit_max = max(6.0, np.ceil(max_freq * 2) / 2)
ax.set_ylim(0, y_limit_max) 
ax.set_ylim(3, 6) # Keeping the original hardcoded limit for now

# Set X-axis
ax.set_xticks(x_positions)
# Applying MODIFICATION 1: Use the simplified labels
# ax.set_xticklabels(new_x_labels, rotation=45, ha='right', fontsize=FONTSIZE-4)
ax.set_xticklabels(new_x_labels, fontsize=FONTSIZE-4)

# MODIFICATION 2: Shorten X-axis label
ax.set_xlabel("Reasoning Progress (%)", fontsize=FONTSIZE)
ax.set_xlim(-0.7, len(frequencies) - 0.3)

# Set Y-axis
# MODIFICATION 3: Shorten Y-axis label
ax.set_ylabel("Switching Frequency (%)", fontsize=FONTSIZE)
# Y-axis formatter for percentage display
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=1))
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=ALLWIDTH/2)
ax.set_axisbelow(True) 

# Adjust layout and save
plt.tight_layout()

# --- Export and Save Plot (as requested) ---
dataset_name = df.iloc[0]['Dataset']
method_name = df.iloc[0]['Method']
output_filename = f'switch_distribution_{dataset_name}_{method_name}_fine_grained_revised.pdf'
plt.savefig(output_filename)

print(f"Chart successfully plotted and saved to: {output_filename}")
print(f"Please check the {output_filename} file in your current directory.")