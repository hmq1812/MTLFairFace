import torch
import argparse
import time
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from agents import FairFaceMultiTaskAgent
from loss import MultiTaskLoss, PseudoLabelingLoss
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on an input image using the trained FairFace Multi-Task Model")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    return parser.parse_args()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def infer(args):
    # Load the image
    image = load_image(args.image_path)

    # Initialize the agent and load the model
    # This loss function is just to fill all neeeded params, not actually in use in inference and eval.
    loss_fn = MultiTaskLoss(task_names=config.CLASS_NAME, loss_weights=config.LOSS_WEIGHT)
    agent = FairFaceMultiTaskAgent(loss_fn, config.CLASS_NAME, config.CLASS_LIST, config.LOSS_WEIGHT)
    agent.load_model(config.MODEL_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.model.to(device)
    image = Variable(image).to(device)


    # Define class names for each task
    age_classes = config.AGE_CLASSES
    gender_classes = config.GENDER_CLASSES
    race_classes = config.RACE_CLASSES
    
    task_class_names = {'age': age_classes, 'gender': gender_classes, 'race': race_classes}


    # Perform inference and measure time
    start_time = time.time()
    with torch.no_grad():
        agent.model.eval()
        outputs = agent.model(image)
    end_time = time.time()

    # Print predictions
    print("Predictions:")
    for task, output in outputs.items():
        prob = torch.nn.functional.softmax(output.cpu(), dim=1)
        prediction = prob.argmax().item()
        confidence = prob[0][prediction].item()

        print(f"{task.capitalize()}: {task_class_names[task][prediction]} (Confidence: {confidence:.4f})")

    print(f"\nInference Time: {end_time - start_time:.4f} seconds")

def main():
    args = parse_args()
    infer(args)

if __name__ == '__main__':
    main()
