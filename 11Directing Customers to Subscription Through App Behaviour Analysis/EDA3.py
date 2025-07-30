import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os

try:
    file_path = 'D:/Desktop/Directing-Customers-to-Subscription-Through-App-Behaviour-Analysis/11Directing Customers to Subscription Through App Behaviour Analysis/appdata10.csv'
    dataset = pd.read_csv(file_path, on_bad_lines='skip')
except FileNotFoundError:
    raise FileNotFoundError(f"'{file_path}' not found. Please check the file path.")

print("\nDataset Preview:")
print(dataset.head(10))
print("\nDataset Description:")
print(dataset.describe())

try:
    dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)
except Exception as e:
    print(f"Hour parsing failed: {e}")

dataset2 = dataset.copy().drop(columns=[
    'user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'
])

plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

dataset2.corrwith(dataset.enrolled).plot.bar(
    figsize=(20, 10),
    title='Correlation with Response Variable',
    fontsize=15,
    rot=45,
    grid=True
)
plt.show()

sn.set(style="white", font_scale=2)
corr = dataset2.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize=40)
cmap = sn.diverging_palette(220, 10, as_cmap=True)

sn.heatmap(
    corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5}
)
plt.show()

try:
    dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
    dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
except Exception as e:
    print(f"Date parsing failed: {e}")

try:
    dataset["difference"] = (dataset.enrolled_date - dataset.first_open).dt.total_seconds() / 3600
except Exception as e:
    print(f"Time difference calculation failed: {e}")

plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

plt.hist(dataset["difference"].dropna(), color='#3F5D7D', range=[0, 100])
plt.title('Distribution of Time-Since-Screen-Reached (0-100 hrs)')
plt.show()

dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset.drop(columns=['first_open', 'enrolled_date', 'difference'], inplace=True)

try:
    top_screens = pd.read_csv('top_screens.csv').top_screens.values
except FileNotFoundError:
    raise FileNotFoundError("'top_screens.csv' not found.")

dataset["screen_list"] = dataset.screen_list.astype(str) + ','

for screen in top_screens:
    dataset[screen] = dataset.screen_list.str.contains(screen).astype(int)
    dataset['screen_list'] = dataset['screen_list'].str.replace(screen + ",", "", regex=False)

dataset['Other'] = dataset.screen_list.str.count(",")
dataset.drop(columns=['screen_list'], inplace=True)

savings = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
credit = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
ccards = ["CC1", "CC1Category", "CC3"]
loans = ["Loan", "Loan2", "Loan3", "Loan4"]

funnel_features = {
    "SavingCount": savings,
    "CMCount": credit,
    "CCCount": ccards,
    "LoansCount": loans
}

for new_col, screens in funnel_features.items():
    dataset[new_col] = dataset[screens].sum(axis=1)
    dataset.drop(columns=screens, inplace=True)

output_file = 'new_appdata10.csv'
dataset.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved as: {output_file}")
