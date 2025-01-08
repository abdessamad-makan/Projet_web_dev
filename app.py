from flask import Flask, render_template, request, url_for, redirect
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import os
import numpy as np

app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# CNN Classifier model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load GAN model
def load_gan_model(model_type):
    model = Generator().to(device)
    model_path = f'models/{model_type.lower()}_generator.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.eval()
    return model

# Load CNN model
def load_cnn_model():
    model = CNNClassifier().to(device)
    model_path = 'models/cnn_classifier.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.eval()
    return model

# Generate images
# Generate images
def generate_images(generator, num_images):
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    fake_images = generator(noise)
    images = []
    for i in range(num_images):
        img = fake_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img + 1) * 127.5).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        # Save image in the 'static' folder
        image_filename = f"generated_image_{i}.jpg"
        image_path = os.path.join("static", image_filename)
        pil_img.save(image_path)
        
        # Store image filename and download link
        images.append({
            'filename': image_filename,
            'download_url': url_for('static', filename=image_filename)
        })
    return images


# Classify image
def classify_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    prediction = output.item()
    return "Normal" if prediction >= 0.5 else "Covid-19"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        model_type = request.form['model_type']
        num_images = int(request.form['num_images'])
        generator = load_gan_model(model_type)
        images = generate_images(generator, num_images)
        return render_template('generate.html', images=images, model_type=model_type)
    return render_template('generate.html', images=None)

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            model = load_cnn_model()
            label = classify_image(model, image)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return render_template('classify.html', label=label, img_str=img_str)
    return render_template('classify.html', label=None)

if __name__ == '__main__':
    app.run(debug=True)
