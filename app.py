import gradio as gr
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics import jaccard_score
from transformers import AutoModelForImageSegmentation

# Load model only once
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
birefnet.to(device)
birefnet.eval()
print(f'BiRefNet loaded on {device}')

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

def process_image(image, ground_truth):
    pred_mask = predict_mask(image)
    
    if ground_truth:
        true_mask = Image.open(ground_truth)
        iou_score = calculate_iou(pred_mask, true_mask)
        iou_text = f"IoU Score: {iou_score:.4f}"
    else:
        iou_text = "No ground truth provided. IoU score cannot be calculated."
    
    # Save prediction to output directory
    if not os.path.exists('data/output'):
        os.makedirs('data/output', exist_ok=True)
    pred_mask.save('data/output/pred_mask.png')
    
    return pred_mask, iou_text

with gr.Blocks() as app:
    gr.Markdown("# Camouflaged Object Detection")
    gr.Markdown("Upload an image to generate a mask and (optionally) upload a ground truth mask to calculate IoU.")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Input Image")
        gt_input = gr.Image(type="filepath", label="Upload Ground Truth Mask (optional)")
    
    predict_button = gr.Button("Generate Prediction")
    output_image = gr.Image(label="Predicted Mask")
    output_text = gr.Text(label="IoU Score")

    predict_button.click(fn=process_image, inputs=[image_input, gt_input], outputs=[output_image, output_text])

app.launch()
