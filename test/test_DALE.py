# Import the required libraries
import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

# Add the root directory of the project to the Python path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# Import project modules
try:
    from model.VisualAttentionNetwork import VisualAttentionNetwork
    from model.EnhancementNet import EnhancementNet
    from train import train_utils
except ImportError:
    print("Error: Couldn't import required modules.")
    print("Please run the script as follows:")
    print("  1. cd ~/Desktop/eklavya/DALE")
    print("  2. source setup_dale.sh")
    print("  3. python test/test_DALE.py")
    sys.exit(1)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Set up paths
    model_root = './checkpoint/'
    test_image_path = './image.jpeg'
    test_result_root_dir = './test_results/'
    
    # Create result directory if needed
    if not os.path.exists(test_result_root_dir):
        os.makedirs(test_result_root_dir)
    
    # Check if model files exist
    if not os.path.exists(model_root + 'VAN.pth'):
        print(f"Error: Model file {model_root + 'VAN.pth'} not found!")
        sys.exit(1)
    if not os.path.exists(model_root + 'EN.pth'):
        print(f"Error: Model file {model_root + 'EN.pth'} not found!")
        sys.exit(1)
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image {test_image_path} not found!")
        sys.exit(1)
    
    # Instantiate and load models
    print("Creating Visual Attention Network...")
    VAN = VisualAttentionNetwork().to(device)
    state_dict1 = torch.load(model_root + 'VAN.pth', map_location=device)
    VAN.load_state_dict(state_dict1)
    
    print("Creating Enhancement Network...")
    EN = EnhancementNet().to(device)
    state_dict2 = torch.load(model_root + 'EN.pth', map_location=device)
    EN.load_state_dict(state_dict2)
    
    # Process image
    print(f"Loading test image: {test_image_path}")
    input_image = Image.open(test_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Run test
    test(input_image_tensor, VAN, EN, test_result_root_dir)

def test(input_image_tensor, VAN, EN, root_dir):
    print("Setting models to evaluation mode...")
    VAN.eval()
    EN.eval()
    
    print("Processing image...")
    with torch.no_grad():
        testImg = input_image_tensor
        
        visual_attention_map = VAN(testImg)
        enhance_result = EN(testImg, visual_attention_map)
        
        enhance_result_img = train_utils.tensor2im(enhance_result)
        
        result_save_dir = os.path.join(root_dir, 'enhanced_image.png')
        train_utils.save_images(enhance_result_img, result_save_dir)
        print(f"Enhanced image saved to {result_save_dir}")

if __name__ == "__main__":
    print("Starting DALE test script...")
    main()
    print("Test completed successfully!")