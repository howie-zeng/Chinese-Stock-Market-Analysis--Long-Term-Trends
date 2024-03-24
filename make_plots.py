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

def feature_importance_boxplot(importance_df):
    normalized_importance = (importance_df / importance_df.sum(axis=0))
    mean_importance_scores = normalized_importance.mean(axis=1)
    sorted_features = mean_importance_scores.sort_values(ascending=False).index
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=normalized_importance.loc[sorted_features].T, orient='h')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.title('Importance Across Models')
    plt.tight_layout()
    plt.show()

def characteristic_feature_importance(importance_df):
    normalized_df = (importance_df / importance_df.sum(axis=0))
    mean_importance_scores = normalized_df.mean(axis=1)
    sorted_features = mean_importance_scores.sort_values(ascending=False).index
    sorted_df = normalized_df.loc[sorted_features]
    plt.figure(figsize=(10, 15))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(sorted_df, cmap=cmap, cbar_kws={"shrink": .82}, linewidths=.5, annot=False, yticklabels=True, xticklabels=True)
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.title('Feature Importance Across Models')
    plt.ylabel('Feature')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.show()
    return sorted_df

def differences_in_feature_importance(first, second):
    # all models
    # different kinds of models
    # NN model
    # different category of feature
    raise NotImplementedError

