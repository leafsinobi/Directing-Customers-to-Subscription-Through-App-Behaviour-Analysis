import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import logging
import os
import sys
from datetime import datetime

# --------------------- Configuration ---------------------
INPUT_FILE = 'D:/Desktop/Directing-Customers-to-Subscription-Through-App-Behaviour-Analysis/11Directing Customers to Subscription Through App Behaviour Analysis/appdata10.csv'
TOP_SCREENS_FILE = 'top_screens.csv'
OUTPUT_FILE_PREFIX = 'new_appdata10'
LOG_FILE = 'eda_processing.log'

# --------------------- Setup Logging ---------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

def log_and_print(message):
    print(message)
    logging.info(message)

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='warn')
        log_and_print(f"Loaded dataset: {file_path} | Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(f"❌ ERROR: File not found → {file_path}")

# --------------------- Visualization Helpers ---------------------
def plot_bar(series, title, filename):
    plt.figure(figsize=(20, 8))
    series.plot.bar(title=title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_histograms(df, filename):
    plt.suptitle('Histograms of Numerical Columns', fontsize=20)
    for i in range(1, df.shape[1] + 1):
        plt.subplot(3, 3, i)
        plt.title(df.columns.values[i - 1])
        vals = np.size(df.iloc[:, i - 1].unique())
        plt.hist(df.iloc[:, i - 1], bins=min(50, vals), color='#3F5D7D')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()

# --------------------- Main EDA Logic ---------------------
def run_eda(input_path, top_screens_path, output_prefix):
    dataset = load_csv(input_path)

    # Log missing values
    log_and_print("\nMissing values per column:")
    log_and_print(dataset.isnull().sum().to_string())

    # Hour extraction
    if 'hour' in dataset.columns:
        try:
            dataset["hour"] = dataset["hour"].astype(str).str.slice(1, 3).astype(float)
            log_and_print("Parsed 'hour' column successfully.")
        except Exception as e:
            logging.warning(f"Hour parsing failed: {e}")

    # Date conversion
    dataset["first_open"] = pd.to_datetime(dataset["first_open"], errors='coerce')
    dataset["enrolled_date"] = pd.to_datetime(dataset["enrolled_date"], errors='coerce')
    dataset["difference"] = (dataset["enrolled_date"].fillna(dataset["first_open"]) - dataset["first_open"]).dt.total_seconds() / 3600
    dataset.loc[dataset["difference"] > 48, 'enrolled'] = 0
    dataset.drop(columns=['first_open', 'enrolled_date', 'difference'], inplace=True)

    # Correlation analysis
    dataset2 = dataset.drop(columns=['user', 'screen_list', 'enrolled'], errors='ignore')
    plot_bar(dataset2.corrwith(dataset['enrolled']), 'Correlation with Enrolled', 'correlation_with_target.png')
    plot_heatmap(dataset2.corr(), 'Correlation Matrix', 'correlation_matrix.png')
    plot_histograms(dataset2, 'feature_histograms.png')

    # Screen list parsing
    dataset["screen_list"] = dataset["screen_list"].astype(str) + ","
    top_screens = load_csv(top_screens_path).top_screens.values
    for screen in top_screens:
        dataset[screen] = dataset["screen_list"].str.contains(screen).astype(int)
        dataset["screen_list"] = dataset["screen_list"].str.replace(screen + ",", "", regex=False)

    dataset["Other"] = dataset["screen_list"].str.count(",")
    dataset.drop(columns=["screen_list"], inplace=True)

    # Funnel aggregation
    funnels = {
        "SavingCount": ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"],
        "CMCount": ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"],
        "CCCount": ["CC1", "CC1Category", "CC3"],
        "LoansCount": ["Loan", "Loan2", "Loan3", "Loan4"]
    }
    for new_feature, screens in funnels.items():
        dataset[new_feature] = dataset[screens].sum(axis=1)
        dataset.drop(columns=screens, inplace=True)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_prefix}_{timestamp}.csv"
    dataset.to_csv(output_file, index=False)
    log_and_print(f"\n✅ Cleaned dataset saved as: {output_file}")

# --------------------- Run ---------------------
run_eda(INPUT_FILE, TOP_SCREENS_FILE, OUTPUT_FILE_PREFIX)
