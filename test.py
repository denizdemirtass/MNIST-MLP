import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# Modeli test edebilmek için mimariyi tekrar tanımlıyoruz
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_my_digit(image_path, model):
    # Görüntüyü aç ve gri tonlamaya çevir
    img = Image.open(image_path).convert('L')
    
    # Beyaz kağıda siyah yazıyı MNIST formatına (siyah arka plan, beyaz yazı) çevir
    img = ImageOps.invert(img)
    
    # 28x28 boyutlandır
    img = img.resize((28, 28))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

# Eğitilen modeli yükle
model = SimpleMLP().to(device)
model.load_state_dict(torch.load("mlp_mnist_model.pth"))

# Kestiğin rakamı test et
print("Tahmin Edilen Rakam:", predict_my_digit("rakam11.jpg", model))