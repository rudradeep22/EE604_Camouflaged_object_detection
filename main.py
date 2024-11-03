import os
from PIL import Image
import torch
from torchvision import transforms
from IPython.display import display
from sklearn.metrics import jaccard_score
import numpy as np
from transformers import AutoModelForImageSegmentation

birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
IMAGE_PATH = 'data/images/COD10K-CAM-2-Terrestrial-37-Lion-2105.jpg'
GROUND_TRUTH_PATH = 'data/gt/COD10K-CAM-2-Terrestrial-37-Lion-2105.png'

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Available GPU:", gpu_name)
else:
    print("CUDA is not available. Using CPU.")

device = input('Enter device (cpu/cuda): ')
print(f'Using device : {device}')
torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_mask(image_path):
    image = Image.open(image_path)        
    input_image = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet(input_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    return pred_pil

def calculate_iou(pred_mask, true_mask):
    resize_transform = transforms.Resize(pred_mask.size, antialias=True)  
    gt_resized = resize_transform(true_mask)
    pred_arr = (np.array(pred_mask) > 0.5).flatten()
    gt_arr = (np.array(gt_resized.convert('L')) > 0).flatten()

    iou_score = jaccard_score(gt_arr, pred_arr)
    return iou_score

pred_mask = predict_mask(IMAGE_PATH)
if GROUND_TRUTH_PATH:
    true_mask = Image.open(GROUND_TRUTH_PATH)
    iou_score = calculate_iou(pred_mask, true_mask)
    print(f'IoU score: {iou_score}')
else:
    print('No ground truth mask provided.')

pred_mask.show()
if os.path.exists('data/output'):
    os.makedirs('data/output', exist_ok=True)
save_path  = 'data/output/pred_mask.png'
pred_mask.save(save_path)
print(f"Image saved to {save_path}")