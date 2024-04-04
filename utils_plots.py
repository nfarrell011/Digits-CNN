"""
    Problem Set 5: Recognition Using Deep Networks
    Joseph Nelson Farrell & Harshil Bhojwani 
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file contains a library of plot utility function related to project 5.

    Functions List (in order):

        1. plot_example_images_with_labels
        2. plot_loss_with_respect_to_samples
        3. plot_accuracy_with_respect_to_samples
        4. plots_results_for_n_images
        5. plot_homebrew_digits
        6. plot_filters_and_impact
        7. plot_filter_weights
        8. plot_greek_accuracy_and_loss
        9. plot_grid_search_results
"""
# import libraries
import matplotlib.pyplot as plt
import numpy as np

###################################################################################################################################################
def plot_example_images_with_labels(example_data, example_target, num_to_plot, save_path, save_name):
    """
        Function: plot_example_images_with_labels
            This function will generate a plot with example images and their true label.

        Parameters:
            example_data: (list) - a list of example images.
            example_target: (list) - corresponding list of target name.
            num_to_plot: (int) - the number of images to plot, up to 6.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    fig = plt.figure()
    fig.suptitle("Example Images with True Labels", weight = 'bold', fontsize = 20)
    for i in range(num_to_plot):
        plt.subplot(2,3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap = "gray", interpolation = 'none')
        plt.title(f"Label: {example_target[i]}", color = 'red', style = 'italic')
        plt.xticks([])
        plt.yticks([])
        plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )

    return None

###################################################################################################################################################
def plot_loss_with_respect_to_samples(test_losses, train_losses, test_counter, train_counter, save_path, save_name):
    """
        Function: plot_loss_with_respect_to_samples
            This function will generate a plot that displays the loss of network training and testing with respect 
            to the number of samples seen.

        Parameters:
            test_losses: (list) - model test losses list.
            train_losses: (list) - model train losses list.
            test_counter: (int) - list of number of samples seen with respect to test losses.
            train_counter: (int) - list of number of samples seen with respect to train losses.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    n_epochs = len(test_losses) - 1
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize = (10, 6))
        plt.plot(train_counter, train_losses, color = 'darkorchid', alpha = .5)
        plt.scatter(test_counter, test_losses, color = 'red')
        for x, y in zip(test_counter, test_losses):
            plt.annotate(f'{y:.3f}',
                        (x, y), 
                        textcoords = "offset points", 
                        xytext = (0, 10), 
                        ha = 'center',
                        color = 'red',
                        fontsize = 7,
                        weight = 'bold') 
        plt.ylim((-.1, 2.6))
        plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
        plt.xlabel('Training Samples Seen', weight = 'bold')
        plt.ylabel('Loss ~ Negative Log Liklihood', weight = 'bold')
        plt.text(0.5, 1.13, 
                'Model Results', 
                fontsize = 18, 
                ha='center', 
                va='bottom', 
                transform = plt.gca().transAxes, 
                weight = 'bold')
        plt.text(0.5, 1.07, 
                'Loss with Respect to Training Samples Seen', 
                fontsize = 12, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes, 
                weight = 'bold',
                style = 'italic')
        plt.text(0.5, 1.01, 
                f'Epochs: {n_epochs}', 
                fontsize = 12, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes, 
                weight = 'bold',
                style = 'italic',
                color = 'orange')
        plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )
        return None
    
###################################################################################################################################################
def plot_accuracy_with_respect_to_samples(test_counter, test_accuracy_list, train_accuracy_list, save_path, save_name):
    """
        Function: plot_accuracy_with_respect_to_samples
            This function will generate a plot that displays the test and train accuracy of a network with respect 
            to the number of samples seen.

        Parameters:
            test_accuracy: (list) - list of test accuracy scores.
            train_losses: (list) - list of train accuracy scores.
            test_counter: (int) - list of number of samples seen with respect to test losses
            train_counter: (int) - list of number of samples seen with respect to train losses
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    n_epochs = len(train_accuracy_list)
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize = (10, 6))
        plt.plot(test_counter, test_accuracy_list, marker = "o", mfc = 'none', color = 'red')
        plt.plot(test_counter[1:], train_accuracy_list, marker = "s", mfc = 'none', color = 'cyan')
        for x, y in zip(test_counter, test_accuracy_list):
            plt.annotate(f'{y:.3f}',
                        (x, y), 
                        textcoords = "offset points", 
                        xytext = (0, 10), 
                        ha = 'center',
                        color = 'red',
                        fontsize = 7,
                        weight = 'bold')
        for x, y in zip(test_counter[1:], train_accuracy_list):
            plt.annotate(f'{y:.3f}',
                        (x, y), 
                        textcoords = "offset points", 
                        xytext = (0, -15), 
                        ha = 'center',
                        color = 'cyan',
                        fontsize = 7,
                        weight = 'bold') 
        plt.legend(['Test Accuracy', 'Train Accuracy'], loc = 'lower right')
        plt.xlabel('Training Samples Seen', weight = 'bold')
        plt.ylabel('Accuracy (%)', weight = 'bold')
        plt.text(0.5, 1.13, 
                'Model Results', 
                fontsize = 18, 
                ha='center', 
                va='bottom', 
                transform = plt.gca().transAxes, 
                weight = 'bold')
        plt.text(0.5, 1.07, 
                'Accuracy Score', 
                fontsize = 12, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes, 
                weight = 'bold',
                style = 'italic')
        plt.text(0.5, 1.01, 
                f'Epochs: {n_epochs}', 
                fontsize = 12, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes, 
                weight = 'bold',
                style = 'italic',
                color = 'orange')
        plt.ylim(0, 110)

        plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )
        return None
    
###################################################################################################################################################
def plots_results_for_n_images(image_data, true_labels, predicted_labels, save_path, save_name):
    """
        Function: plot_results_for_n_images
            This function will generate a plot with example images and their true and predicted labels.

        Parameters:
            image_data: (list) - a list of example images.
            true_labels: (list) - corresponding list of target name.
            predicted_labels: (list) - corresponding list of predicted labels.
            num_to_plot: (int) - the number of images to plot, up to 6.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    fig = plt.figure(figsize = (10, 10))
    fig.suptitle(f'True versus Predicted Labels \n First 9 Digits', weight = "bold", fontsize = 20)
    for i in range(9):
        plt.subplot(3,3, i + 1)
        plt.tight_layout()
        plt.imshow(image_data[i][0], cmap = "gray", interpolation = 'none')
        plt.text(0.5, 1.10, 
            f'True Label: {true_labels[i]}', 
            fontsize = 10, 
            ha='center', 
            va='bottom', 
            transform = plt.gca().transAxes,
            color = 'blue',
            weight = 'bold')
        plt.text(0.5, 1.02, 
            f'Predicted Label: {predicted_labels[i]}', 
            fontsize = 10, 
            ha = 'center', 
            va = 'bottom', 
            transform = plt.gca().transAxes,
            color = "red",
            weight = 'bold',
            style = 'italic')
        plt.xticks([])
        plt.yticks([])
    plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )
    return None

###################################################################################################################################################
def plot_homebrew_digits(image_list, true_label_list, predicted_labels, save_path, save_name):
    """
        Function: plot_home_brew_digits
            This function will generate a plot with example homebrew digit images and their true and predicted labels.

        Parameters:
            image_data: (list) - a list of example images.
            true_labels_list: (list) - corresponding list of target name.
            predicted_labels: (list) - corresponding list of predicted labels.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    fig = plt.figure(figsize = (10, 10))
    fig.suptitle(f'True versus Predicted Labels \nHand Drawn Digits', weight = "bold", fontsize = 20)
    for i in range(len(image_list)):
        plt.subplot(3,4, i + 1)
        plt.tight_layout()
        plt.imshow(image_list[i], cmap = "gray", interpolation = 'none')
        plt.text(0.5, 1.10, 
            f'True Label: {true_label_list[i]}', 
            fontsize = 10, 
            ha='center', 
            va='bottom', 
            transform = plt.gca().transAxes,
            color = 'blue',
            weight = 'bold')
        plt.text(0.5, 1.02, 
            f'Predicted Label: {predicted_labels[i]}', 
            fontsize = 10, 
            ha = 'center', 
            va = 'bottom', 
            transform = plt.gca().transAxes,
            color = "red",
            weight = 'bold',
            style = 'italic')
        plt.xticks([])
        plt.yticks([])
    plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )
    return None

###################################################################################################################################################
def plot_filter_weights(layer, save_path, save_name):
    """
        Function: plot_fitler_weights
            This function will generate a plot of a layer of a neural networks filter weights
            as a heatmap

        Parameters:
            layer: (list) - a list of matrices, filter weights.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(3, 4, figsize = (10, 10))
        fig.suptitle(f'Layer 1 Filter Weights', weight = 'bold', fontsize = 20)
        axes = axes.ravel()
        for i in range(10):
            ax = axes[i]
            ax.imshow(layer[i], cmap = 'Reds')  # You can choose a colormap suitable for your filters
            ax.set_title(f"Filter: {i}", weight = 'bold', style = "italic")
            ax.set_xticks([])
            ax.set_yticks([])
        for i in range(10, 12):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )

    return None

###################################################################################################################################################
def plot_filters_and_impact(filters, filtered_images, save_path, save_name):
    """
        Function: plot_fitler_weights
            This function will generate a plot of a layer of a neural networks filter weights
            as a heatmap, and their corresponding impact on an image.

        Parameters:
            layer: (list) - a list of matrices, filter weights.
            fitlered_images: (list) - list of images after they have been filtered.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(5, 4, figsize = (10, 10))
        plt.suptitle(f'Layer 1 Filters & Their Impact', weight = 'bold', fontsize = 20)
        for i in range(5):
            for j in range(4):
                if j % 2 == 0:
                    axes[i, j].imshow(filters[i], cmap = "Blues")
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                else:
                    axes[i, j].imshow(filtered_images[i], "Blues")
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
        plt.tight_layout()
        plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight' )
    
    return None

###################################################################################################################################################
def plot_greek_accuracy_and_loss(epochs_for_x_axis, train_accuracy_list, train_losses_list, title_suffix, save_path, save_name):
    """
        Function: plot_greek_accuracy_and_loss
            This function will generate a plot that displays the test and train accuracy and the loss of a network with respect 
            to the number of epochs.

        Parameters:
            epochs_for_x_axis: (list) - a list of the number of epochs.
            train_accuracy_list: (list) - list of test accuracy scores.
            train_losses_list: (list) - list of train accuracy scores.
            title_suffix: (str) - a string to add to plot title
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(2, 1, figsize = (10, 8), sharex = True)
        fig.suptitle(f'Model Results ~ {title_suffix}', 
                    fontsize = 18, 
                    weight='bold', 
                    va = 'top', 
                    ha = 'center')
        fig.subplots_adjust(top = 1)

        # accuracy
        axes[0].plot(epochs_for_x_axis, train_accuracy_list, color='cyan', alpha=.5)
        for x, y in zip(epochs_for_x_axis, train_accuracy_list):
            axes[0].annotate(f'{y:.1f}', 
                            (x, y), 
                            textcoords = "offset points", 
                            xytext = (0, 10), 
                            ha = 'center', 
                            color = 'cyan', 
                            fontsize = 7, 
                            weight = 'bold')
        axes[0].set_ylabel('Accuracy (%)', weight='bold', color = 'cyan')
        axes[0].set_title(f'Accuracy with Respect Epochs', 
                        weight = 'bold', 
                        style = 'italic', 
                        fontsize = 10, 
                        color = 'cyan')
        axes[0].set_ylim(0, 110)

        # losses
        axes[1].plot(epochs_for_x_axis, train_losses_list, color = 'red', alpha=.5)
        for x, y in zip(epochs_for_x_axis, train_losses_list):
            axes[1].annotate(f'{y:.3f}', 
                            (x, y), 
                            textcoords = "offset points", 
                            xytext = (0, 10), 
                            ha = 'center', 
                            color = 'red', 
                            fontsize = 7, 
                            weight = 'bold')
        axes[1].set_title(f'Loss with Respect Epochs', 
                        weight = 'bold', 
                        style = 'italic', 
                        fontsize = 10, 
                        color = 'red')
        axes[1].set_ylabel('Loss', weight='bold', color = 'red')
        axes[1].set_xlabel('Epochs', weight='bold')
        axes[1].set_ylim(0, 2.8)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig( save_path + '/figs/' + save_name, bbox_inches = 'tight')
    return None

###################################################################################################################################################
def plot_grid_search_results(results_frame, batches, save_path, save_name):
    """
        Function: plot_grid_search_results
            This function will plot the accuracy score of each model that was trained in a gridsearch.

        Parameters:
            results_frame: (pd.DataFrame) - dataframe containing grid search results.
            save_path: (str) - path to the current directory.
            save_name: (str) - name of plot.

        Returns: 
            None.
    """
    #models_list = ["Model 1", "Model 2", "Model 3"]
    #batches = [32, 64, 128]
    cols = results_frame.columns
    i = 0
    for model_ in cols:
        model = results_frame[model_]
        for batch in batches:
            test_accuracy_list = model.loc[batch]['test_accuracy']
            train_accuracy_list = model.loc[batch]['train_accuracy']
            test_counter = model.loc[batch]['test_counter']
            epochs = np.arange(0, len(test_counter), 1)
            with plt.style.context('dark_background'):
                fig = plt.figure(figsize = (10, 6))
                plt.plot(epochs, test_accuracy_list, marker = "o", mfc = 'none', color = 'red')
                plt.plot(epochs[1:], train_accuracy_list, marker = "s", mfc = 'none', color = 'cyan')
                for x, y in zip(epochs, test_accuracy_list):
                    plt.annotate(f'{y:.3f}',
                                (x, y), 
                                textcoords = "offset points", 
                                xytext = (0, 10), 
                                ha = 'center',
                                color = 'red',
                                fontsize = 7,
                                weight = 'bold')
                for x, y in zip(epochs[1:], train_accuracy_list):
                    plt.annotate(f'{y:.3f}',
                                (x, y), 
                                textcoords = "offset points", 
                                xytext = (0, -15), 
                                ha = 'center',
                                color = 'cyan',
                                fontsize = 7,
                                weight = 'bold') 
                plt.legend(['Test Accuracy', 'Train Accuracy'], loc = 'lower right')
                plt.xlabel('Epochs', weight = 'bold')
                plt.ylabel('Accuracy (%)', weight = 'bold')
                plt.text(0.5, 1.13, 
                        f'Model {i + 1}', 
                        fontsize = 18, 
                        ha='center', 
                        va='bottom', 
                        transform = plt.gca().transAxes, 
                        weight = 'bold')
                plt.text(0.5, 1.07, 
                        'Accuracy Score', 
                        fontsize = 12, 
                        ha = 'center', 
                        va = 'bottom', 
                        transform = plt.gca().transAxes, 
                        weight = 'bold',
                        style = 'italic')
                plt.text(0.5, 1.01, 
                        f'Batch Size: {batch}', 
                        fontsize = 12, 
                        ha = 'center', 
                        va = 'bottom', 
                        transform = plt.gca().transAxes, 
                        weight = 'bold',
                        style = 'italic',
                        color = 'orange')
                batch_str = str(batch)
                save_name_ = model_ + "_" + batch_str + "_" + save_name
                plt.ylim(0, 110)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig( save_path + '/figs/' + save_name_, bbox_inches = 'tight')
        i += 1

    return None