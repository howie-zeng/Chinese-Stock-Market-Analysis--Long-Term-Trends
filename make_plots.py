import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_values(data):
    missing_values_proportion = data.isna().mean() * 100

    plt.figure(figsize=(10, 6))
    plt.hist(missing_values_proportion)
    plt.title('Histogram of Proportion of Missing Values')
    plt.xlabel('Proportion of Missing Values in %')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def macroeconomic_feature_importance(percent_change_df):
    mean_percent_change = percent_change_df.mean(axis=1)
    sorted_features = mean_percent_change.sort_values(ascending=False).index
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=percent_change_df.loc[sorted_features].T, orient='h')
    plt.xlabel('Percentage Change% in R²')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def characteristic_feature_importance(importance_df):
    normalized_df = importance_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    cmap = sns.color_palette("Blues", as_cmap=True)
    sorted_features = normalized_df.mean(axis=1).sort_values(ascending=False).index
    sorted_df = normalized_df.loc[sorted_features]
    plt.figure(figsize=(10, 15))
    sns.heatmap(sorted_df, cmap=cmap, cbar_kws={"shrink": .82}, linewidths=.5, annot=False, yticklabels=True, xticklabels=True)
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.title('Feature Importance Across Models')
    plt.ylabel('Feature')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.show()

    return sorted_df

def differences_in_feature_importance(first, second):
    raise NotImplementedError

