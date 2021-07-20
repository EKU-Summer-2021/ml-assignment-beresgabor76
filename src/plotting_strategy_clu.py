"""
Module for results' PlottingStrategy for datasets of Classification
"""
import os
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from src.plotting_strategy import PlottingStrategy


class PlottingStrategy4CLU(PlottingStrategy):
    """
    Results' plotting strategy implementation for Clustering
    """
    def plot_clusters(self, test_set_x, prediction, save_dir):
        """
        Plots out the clusters as scatter plot
        """
        df_dataset = pd.concat([test_set_x, prediction], axis=1)
        n_components = 4
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(df_dataset)
        total_var = pca.explained_variance_ratio_.sum() * 100
        labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
        labels['color'] = 'Label'
        fig = px.scatter_matrix(
            components,
            color=df_dataset['Label'],
            dimensions=range(n_components),
            labels=labels,
            title=f'Total Explained Variance: {total_var:.2f}%',
        )
        fig.update_traces(diagonal_visible=False)
        plot_file = os.path.join(os.path.dirname(__file__), save_dir, 'clusters.png')
        fig.write_image(plot_file)

    def plot_show(self, test_set_y, prediction):
        """
        Shows plot results of learning algorithm's test
        """
