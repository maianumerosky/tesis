import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import pandas as pd
import seaborn as sns
from clasificacion_humedales.utils.utils_maia import directories

PATH_IN, PATH_OUT = directories()

plt.rcParams.update({'font.size': 18})

class Signatures:

    # Esperamos un clustering de Z^n con -3000 como valor de NaN
    # Esperamos imágenes de R^dxn
    # Esperamos eje x de string^n

    def __init__(self, images, x_axis, clustering, name_x_axis, name_y_axis):
        self._images = images
        self._x_axis = x_axis
        self._clustering = clustering
        self._name_x_axis = name_x_axis
        self._name_y_axis = name_y_axis


    def clustering(self):
        return self._clustering


    def images(self):
        return self._images


    def x_axis(self):
        return self._x_axis


    def name_y_axis(self):
        return self._name_y_axis


    def name_x_axis(self):
        return self._name_x_axis


    def create_for(self, class_i, statistic=np.mean):
        signatures = []
        for image in self.images():
            image_cluster_i = image[self.clustering() == class_i]
            signatures.append(statistic(image_cluster_i))
        return np.array(signatures)


    def compare(self, class_1, class_2):
        return np.trapz(abs(self.create_for(class_1, np.mean)-self.create_for(class_2, np.mean)))


    def plot_for_mean(self, classes, save_plot=False, prefix = ""):
        rows = []
        for clustering_i in classes:
            for index, x in enumerate(self.x_axis()):
                image = self.images()[index]
                image_cluster_i = image[self.clustering() == clustering_i]
                for pixel in image_cluster_i:
                    rows.append([x, pixel, clustering_i])

        df_plot = pd.DataFrame(rows, columns=[self.name_x_axis(), self.name_y_axis(), "Clase"])

        palette = sns.color_palette("husl", len(classes))
        fig, ax = plt.subplots(1, 1, figsize=(15, 7));
        ax.set_ylim(-0.21, 1);
        ax.set_xlim(df_plot[f'{self.name_x_axis()}'].min(), df_plot[f'{self.name_x_axis()}'].max());
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y"));
        ax.xaxis.set_major_locator(plt.LinearLocator(numticks=19));
        plt.xticks(rotation = 45);
        plot = sns.lineplot(x=self.name_x_axis(), y=self.name_y_axis(), ci='sd', data=df_plot, hue="Clase",
                            sort=False, palette=palette, ax=ax)
        
        if save_plot:
            plot.figure.savefig(PATH_OUT + f'{prefix}_mean_{"_".join(map(str, classes))}', bbox_inches='tight')
        
        return plot

    def plot_all_mean(self):
        labels = np.delete(np.unique(self.clustering()), 0)
        for label in labels:
            plt.cla()
            plot = self.plot_for_mean(label).get_figure()
            plot.savefig(PATH_OUT + f'plot_cluster_{label}.png')
