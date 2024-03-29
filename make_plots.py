import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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

def plot_model_performance(results, params_to_plot):
    filtered_params = [p for p in params_to_plot if p not in ['max_iter', 'n_jobs']]
    num_params = len(filtered_params)
    nrows = int(num_params ** 0.5)
    ncols = int(num_params / nrows) + (num_params % nrows > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if nrows * ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Encapsulate it in a list if it's a single subplot for consistency
    for i, param_name in enumerate(filtered_params):
        log = ''
        if results[param_name].mean() <= 1:
            results[param_name] = np.log(results[param_name])
            log = "log"
        results_grouped = results.groupby(param_name).mean()
        train_scores = results_grouped['train_score']
        val_scores = results_grouped['val_score']
        param_values = results_grouped.index
        
        ax = axes[i]
        ax.plot(param_values, train_scores, marker='o', linestyle='-', label='Train Score', color='tab:blue')
        ax.plot(param_values, val_scores, marker='s', linestyle='--', label='Validation Score', color='tab:red')
        ax.set_xlabel(f'{param_name}_{log}')
        ax.set_ylabel('Score')
        ax.legend()
        ax.set_title(f'Model Performance by {param_name}')
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

