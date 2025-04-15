import seaborn as sns
import matplotlib.pyplot as plt



def plot_pairplots(df):

    df1 = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Outcome']]
    df2 = df[['Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
    sns.pairplot(df1, hue='Outcome')
    sns.pairplot(df2, hue='Outcome')
    plt.show()



def print_statistics(df):

    attributes = df.columns.drop('Outcome')

    for attr in attributes:

        mean_val = df[attr].mean()
        std_val = df[attr].std()
        min_val = df[attr].min()
        max_val = df[attr].max()
        median_val = df[attr].median()
        print(f"{attr} statistics:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Standard Deviation: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Median: {median_val:.4f}\n")
