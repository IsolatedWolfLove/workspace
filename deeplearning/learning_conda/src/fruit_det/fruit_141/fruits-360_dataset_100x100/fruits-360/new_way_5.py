import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import json
import os

datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/Training"
txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/label_1.txt"

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_map = {
    "Cocos 1": 0,
    "Apple 6": 1,
    "Apple Golden 3": 2,
    "Zucchini 1": 3,
    "Granadilla 1": 4,
    "Eggplant 1": 5,
    "Physalis 1": 6,
    "Banana 1": 7,
    "Peach Flat 1": 8,
    "Tamarillo 1": 9,
    "Huckleberry 1": 10,
    "Apple Crimson Snow 1": 11,
    "Carambula 1": 12,
    "Eggplant long 1": 13,
    "Physalis with Husk 1": 14,
    "Pear Kaiser 1": 15,
    "Pear 2": 16,
    "Strawberry Wedge 1": 17,
    "Kiwi 1": 18,
    "Walnut 1": 19,
    "Maracuja 1": 20,
    "Carrot 1": 21,
    "Potato Sweet 1": 22,
    "Guava 1": 23,
    "Mango 1": 24,
    "Pear Abate 1": 25,
    "Strawberry 1": 26,
    "Avocado 1": 27,
    "Apple hit 1": 28,
    "Kaki 1": 29,
    "Grape White 4": 30,
    "Pineapple 1": 31,
    "Grape White 2": 32,
    "Grape Blue 1": 33,
    "Melon Piel de Sapo 1": 34,
    "Watermelon 1": 35,
    "Tomato 2": 36,
    "Tomato not Ripened 1": 37,
    "Corn Husk 1": 38,
    "Onion Red Peeled 1": 39,
    "Apple Braeburn 1": 40,
    "Pepino 1": 41,
    "Mango Red 1": 42,
    "Kumquats 1": 43,
    "Corn 1": 44,
    "Pomelo Sweetie 1": 45,
    "Rambutan 1": 46,
    "Chestnut 1": 47,
    "Grapefruit Pink 1": 48,
    "Pitahaya Red 1": 49,
    "Onion Red 1": 50,
    "Apple Red 1": 51,
    "Kohlrabi 1": 52,
    "Tomato Maroon 1": 53,
    "Cabbage white 1": 54,
    "Plum 2": 55,
    "Nut Forest 1": 56,
    "Cherry Rainier 1": 57,
    "Lemon Meyer 1": 58,
    "Pepper Green 1": 59,
    "Tomato Heart 1": 60,
    "Pepper Yellow 1": 61,
    "Salak 1": 62,
    "Banana Red 1": 63,
    "Pomegranate 1": 64,
    "Tomato Yellow 1": 65,
    "Cucumber Ripe 2": 66,
    "Potato Red 1": 67,
    "Apple Red 2": 68,
    "Pear Stone 1": 69,
    "Ginger Root 1": 70,
    "Cactus fruit 1": 71,
    "Apple Red 3": 72,
    "Quince 1": 73,
    "Grape White 1": 74,
    "Cherry 1": 75,
    "Cucumber 3": 76,
    "Lychee 1": 77,
    "Cherry 2": 78,
    "Redcurrant 1": 79,
    "Apricot 1": 80,
    "Banana Lady Finger 1": 81,
    "Cauliflower 1": 82,
    "Cherry Wax Red 1": 83,
    "Apple Red Delicious 1": 84,
    "Passion Fruit 1": 85,
    "Hazelnut 1": 86,
    "Papaya 1": 87,
    "Tomato 3": 88,
    "Fig 1": 89,
    "Mangostan 1": 90,
    "Potato White 1": 91,
    "Avocado ripe 1": 92,
    "Apple Golden 1": 93,
    "Peach 1": 94,
    "Apple Red Yellow 2": 95,
    "Limes 1": 96,
    "Cherry Wax Yellow 1": 97,
    "Tangelo 1": 98,
    "Lemon 1": 99,
    "Cherry Wax Black 1": 100,
    "Orange 1": 101,
    "Onion White 1": 102,
    "Plum 1": 103,
    "Pineapple Mini 1": 104,
    "Pear Williams 1": 105,
    "Zucchini dark 1": 106,
    "Raspberry 1": 107,
    "Pepper Red 1": 108,
    "Tomato 1": 109,
    "Dates 1": 110,
    "Pear 1": 111,
    "Grape Pink 1": 112,
    "Cantaloupe 2": 113,
    "Mandarine 1": 114,
    "Grape White 3": 115,
    "Grapefruit White 1": 116,
    "Nut Pecan 1": 117,
    "Apple Golden 2": 118,
    "Apple Pink Lady 1": 119,
    "Peach 2": 120,
    "Tomato 4": 121,
    "Nectarine Flat 1": 122,
    "Plum 3": 123,
    "Tomato Cherry Red 1": 124,
    "Pear Red 1": 125,
    "Blueberry 1": 126,
    "Cucumber Ripe 1": 127,
    "Pear Monster 1": 128,
    "Pear Forelle 1": 129,
    "Cucumber 1": 130,
    "Beetroot 1": 131,
    "Apple Red Yellow 1": 132,
    "Pear 3": 133,
    "Apple Granny Smith 1": 134,
    "Pepper Orange 1": 135,
    "Cantaloupe 1": 136,
    "Mulberry 1": 137,
    "Nectarine 1": 138,
    "Potato Red Washed 1": 139,
    "Clementine 1": 140
}

with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

with open('label_map.json', 'r') as f:
    label_map = json.load(f)

class FlowerDataset(Dataset):
    def __init__(self, txtpath, datapath, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.data = []
        self.datapath = datapath
        self.class_to_idx = label_map
        
        with open(txtpath, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    img_name, label = line.strip().split(',')
                    img_path = os.path.join(datapath, img_name)
                    label = label_map.get(label, -1)
                    if 0 <= label < 141:
                        self.data.append((img_path, label))
                    else:
                        print(f"Warning: Invalid label {label} found in {img_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes=141):
        super().__init__()
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

        scheduler.step()

    torch.save(model.state_dict(), 'mobilenetv3_fruit_classifier.pth')

def load_model(model_path, num_classes=141):
    model = MobileNetV3Classifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        confidence = torch.max(probabilities).item() * 100
        predicted_class = list(label_map.keys())[list(label_map.values()).index(predicted.item())]
    return predicted_class, confidence, image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FlowerDataset(txtpath=txtpath, datapath=datapath, transform=transform_train, train=True)
    val_dataset = FlowerDataset(txtpath=txtpath, datapath=datapath, transform=transform_val, train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = MobileNetV3Classifier(num_classes=141)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    model = load_model('mobilenetv3_fruit_classifier.pth', num_classes=141)
    model.to(device)
    image_path = 'path_to_test_image.jpg'
    predicted_class, confidence, image = predict_image(image_path, model, device)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()