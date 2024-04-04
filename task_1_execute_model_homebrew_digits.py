"""
    Problem Set 5: Recognition Using Deep Networks: Task 1
    Joseph Nelson Farrell
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file will read in the saved network and test it on hand drawn digits.
    It will also process the images to testing them on the model
"""

# functions are in seperate files
from utils_image_processing import process_images
from utils_image_processing import get_processed_images
from utils_plots import plot_homebrew_digits
from utils import get_saved_model

# import packages
import torch
import os
import torch.optim as opt
import sys

##################################################### Main ####################################################################
# main
def main(argv):
    """
        Fucntion: main
            This function will read in the saved network and test it on hand drawn digits. It will also 
            process the images before testing them on the model. It will generate figures of example images, and
            model results.
    
        Parameters: 
            None

        Returns:
            None
    """
    # set parameters
    learning_rate = 0.01
    momentum = 0.5
    random_seed = 42
    model_file = 'results/model.pth'
    optimizer_file = 'results/optimizer.pth'
    raw_image_folder = 'digits_homebrew_raw'
    curated_image_folder = 'digits_homebrew_curated'

    # set torch settings
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # set the current path
    path = os.getcwd()

    # check if the raw images exist
    if not os.path.exists(raw_image_folder):
        raise FileNotFoundError(f'Folder with images to process does not exsit!')

    # check if there is a curated folder, if not make one
    if not os.path.exists(curated_image_folder):
        os.makedirs(curated_image_folder, exist_ok=True)
        print(f"The folder '{curated_image_folder}' was created.")
    else:
        print(f"The folder '{curated_image_folder}' already exists.")

    # check if the figs folder exists
    figs_folder = 'figs'
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok=True)
        print(f"The folder '{figs_folder}' was created.")
    else:
        print(f"The folder '{figs_folder}' already exists.")

    # process the images in the raw folder
    process_images(raw_image_folder, curated_image_folder)

    # set path to curated images
    curated_image_path = os.path.join(path, curated_image_folder)

    # get the processed images
    image_list, true_label_list= get_processed_images(curated_image_path)

    # get the model
    model_path = os.path.join(path, model_file)
    optimizer_path = os.path.join(path, optimizer_file)

    # get the model
    model, optimizer = get_saved_model(model_path, optimizer_path, learning_rate, momentum)

    optimizer = opt.SGD(model.parameters(),
                            lr=learning_rate, momentum=momentum)

    predicted_labels = []
    for i, j in zip(image_list, true_label_list):
        i = (i - 0.1307)/0.3081
        tensor_image = torch.tensor(i, dtype = torch.float32)
        tensor_image = tensor_image.unsqueeze(0)

        # set to no gradient
        with torch.no_grad():
            output = model(tensor_image)
            print('*' * 50)
            probabilities = torch.softmax(output, dim = 1)
            predicted_class = torch.argmax(probabilities, dim = 1)
            predicted_label = predicted_class.item()
            predicted_labels.append(predicted_label)
            print(f"True Class: {j}")
            print(f"Predicted class (index): {predicted_label}")
            print(f'Class Probabilities: {output[0]}')

    # plot results
    save_name = "homebrew_digits_results.png"
    plot_homebrew_digits(image_list, true_label_list, predicted_labels, path, save_name)

if __name__ == "__main__":
    main(sys.argv)


