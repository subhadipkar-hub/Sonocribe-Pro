import torch
from torchvision import transforms
from PIL import Image
from model import build_model

def predict_image(img_path, model_path="breast_cancer_model_v2.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['benign', 'malignant', 'normal']

    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {conf.item()*100:.2f}%")
