import numpy as np
from typing import Dict


def get_data_info(data_set: str,
                  normalize: bool = False) -> Dict:
    """
    returns information (class names, image shape, ...) about data set as a dictionary
    """
    n_bits = 8
    temp = 1
    num_classes = 10
    class_names = None
    if data_set == 'MNIST':
        data_shape = [1, 28, 28]
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        data_mean = np.array([0.1307])
        data_std = np.array([0.3081])
    if data_set == "CelebA":
        class_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
                       "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
                       "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
                       "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
                       "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
                       "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
                       "Wearing_Necktie", "Young"]
        num_classes = 40
        data_shape = [3, 64, 64]
        n_bits = 5
        temp = 0.7
        data_mean = np.array([0.485, 0.456, 0.406])
        data_std = np.array([0.229, 0.224, 0.225])

    if data_set == "CheXpert":
        class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        num_classes = 14
        data_shape = [1, 128, 128]

        data_mean = np.array([0.5412])
        data_std = np.array([0.2777])

    if data_set == "Mall":
        class_names = None
        num_classes = 0
        data_shape = [3, 64, 64]
        n_bits = 5

        data_mean = np.array([0.5])
        data_std = np.array([0.25])

    if not normalize:
        data_mean = np.zeros_like(data_mean)
        data_std = np.ones_like(data_std)

    data_info = {'data_set': data_set, 'data_shape': data_shape, 'n_bits': n_bits, 'temp': temp,
                 'num_classes': num_classes, 'class_names': class_names, 'data_mean': data_mean, 'data_std': data_std}
    return data_info
