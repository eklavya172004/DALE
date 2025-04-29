import sys
import os
# Add the root directory of the project to the Python path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from model import VisualAttentionNetwork, EnhancementNet
from data import dataset_DALE
from train import train_utils
import torch
from torch.utils.data import DataLoader

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    benchmark = ['datasets__DICM', 'datasets__LIME', 'datasets__MEF', 'datasets__NPE', 'dark_face']
    test_data_root = '/home/snu/Desktop/eklavya/DALE/LIME/'+benchmark[4]
    model_root = './checkpoint/'
    test_result_root_dir = './EN_TEST/'

    # Instantiate models and transfer them to the device (GPU or CPU)
    VAN = VisualAttentionNetwork.VisualAttentionNetwork().to(device)
    state_dict1 = torch.load(model_root + 'VAN.pth', map_location=device)
    VAN.load_state_dict(state_dict1)

    EN = EnhancementNet.EnhancementNet().to(device)
    state_dict2 = torch.load(model_root + 'EN.pth', map_location=device)
    EN.load_state_dict(state_dict2)

    # Load test dataset and prepare the DataLoader
    test_data = dataset_DALE.DALETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

    # Move models to evaluation mode
    VAN.eval()
    EN.eval()

    # Run the test function
    test(loader_test, VAN, EN, test_result_root_dir)


def test(loader_test, VAN, EN, root_dir):
    for itr, data in enumerate(loader_test):
        # Load and move data to the GPU
        testImg, img_name = data[0].to(device), data[1]
        
        with torch.no_grad():
            # Forward pass through the models
            visual_attention_map = VAN(testImg)
            enhance_result = EN(testImg, visual_attention_map)
            
            # Convert tensor to image and save the result
            enhance_result_img = train_utils.tensor2im(enhance_result)
            result_save_dir = os.path.join(root_dir, 'enhance' + img_name[0].split('.')[0] + '.jpg')
            train_utils.save_images(enhance_result_img, result_save_dir)

if __name__ == "__main__":
    main()
