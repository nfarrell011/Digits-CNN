"""
    Problem Set 5: Recognition Using Deep Networks: Task 2
    Joseph Nelson Farrell
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file will examine the first later NeuralNetwork.
"""

# functions are in seperate files
from utils import get_saved_model
from utils import get_load_digits_data
from utils import apply_layer_1_filters
from utils_plots import plot_filter_weights
from utils_plots import plot_filters_and_impact

# import packages
import torch
import os
import sys

##################################################### Main ####################################################################
# main
def main(argv):
    """ 
        Function: main
            This function print out model information and generate figures of the first layer.

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
    model_file = 'results/model.pth'
    optimizer_file = 'results/optimizer.pth'


    # set torch settings
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # set the current path
    path = os.getcwd()

    # get the model
    model, _ = get_saved_model(model_file, optimizer_file, learning_rate, momentum)

    # get the digits data
    _, test_loader = get_load_digits_data(batch_size_train, batch_size_test)

    # display model
    print()
    print(f'Model: \n')
    print(model)
    print()

    # get the first layer
    layer_1 = model.conv1.weight

    # convert filters to numpy
    filters = layer_1.data.detach().numpy().squeeze()

    # get the number of filters
    num_filters = layer_1.shape[0]

    # display info to terminal
    print("-" * 75)
    for i in range(num_filters):
        print(f"Filter: {i}")
        print()
        print(f"Shape: {filters[i].shape}")
        print()
        print(f'Weights:')
        print()
        print(f'{filters[i]}')
        print()
        print("-" * 75)

    # generate plot
    save_name = "layer_1_filter_weights.png"
    plot_filter_weights(filters, path, save_name)

    # get the first image
    examples = enumerate(test_loader)
    _, (example_data, _) = next(examples)
    first_image = example_data[0].detach().numpy().squeeze()

    # apply the first layer filters
    filtered_images = apply_layer_1_filters(first_image, filters)

    # generate and save fig
    save_name = "filters_and_filtered_images.png"
    plot_filters_and_impact(filters, filtered_images, path, save_name)

if __name__ == "__main__":
    main(sys.argv)





