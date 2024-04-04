"""
    Problem Set 5: Recognition Using Deep Networks: Task 1
    Joseph Nelson Farrell & Harshil Bhojwani 
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file will test the trained NeualNetwork on the first 10 digits in the digits dataset.
"""
# functions are in seperate files, utils and utils_plots
from utils import get_load_digits_data
from utils import get_saved_model
from utils import execute_model_n_times
from utils_plots import plots_results_for_n_images

# import packages
import torch
import sys
import os

############################################### Main ########################################################################
# main 
def main(argv):
    """
        Function: main
            This function will test the trained model on the first 10 digits in the digits dataset

        Parameters: 
            None

        Returns:
            None
    """
    # set parameters
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    random_seed = 42
    num_images_test = 10
    model_file = 'results/model.pth'
    optimizer_file = 'results/optimizer.pth'

    # set torch settings
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # set the current path
    path = os.getcwd()

    # check if the figs folder exists
    figs_folder = 'figs'
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok=True)
        print(f"The folder '{figs_folder}' was created.")
    else:
        print(f"The folder '{figs_folder}' already exists.")

    model_path = os.path.join(path, model_file)
    optimizer_path = os.path.join(path, optimizer_file)

    # get the data
    _, test_loader = get_load_digits_data(batch_size_train, batch_size_test)

    # get the model
    model, _ = get_saved_model(model_path, optimizer_path, learning_rate, momentum)

    # execute model on the first 10 images
    true_labels, predicted_labels, _ = execute_model_n_times(model, test_loader, num_images_test)

    # get the iamge data
    examples = enumerate(test_loader)
    _, (example_data, _) = next(examples)

    # name of file to save plot
    save_name = "first_10_image_results.png"

    # plot results
    plots_results_for_n_images(example_data, true_labels, predicted_labels, path, save_name)

if __name__ == "__main__":
    main(sys.argv)
