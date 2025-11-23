import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
import io
import base64
from PIL import Image, ImageOps
from torchvision import transforms

app = Flask(__name__)


# --- ОПРЕДЕЛЕНИЕ КЛАССОВ (Обязательно такие же, как при обучении) ---

# 1. Логистическая регрессия (из твоего файла mnist-logistic.pth)
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out


# 2. MLP (из mnist-mlp.pth)
class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, xb):
        return self.network(xb)


# 3. CNN (из mnist-cnn.pth)
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, xb):
        return self.network(xb)


# --- ЗАГРУЗКА МОДЕЛЕЙ ---
# Создаем экземпляры
model_logistic = MnistModel()
model_mlp = MnistMLP()
model_cnn = MnistCNN()

# Грузим веса (map_location='cpu' важно, если обучал на GPU, а запускаешь на CPU)
try:
    model_logistic.load_state_dict(torch.load('models/mnist-logistic.pth', map_location='cpu'))
    model_mlp.load_state_dict(torch.load('models/mnist-mlp.pth', map_location='cpu'))
    model_cnn.load_state_dict(torch.load('models/mnist-cnn.pth', map_location='cpu'))
    print("Все модели успешно загружены!")
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")

# Переводим в режим предсказания (evaluation mode)
model_logistic.eval()
model_mlp.eval()
model_cnn.eval()


# --- ОБРАБОТКА ---
def transform_image(image_bytes):
    # Декодируем base64
    image = Image.open(io.BytesIO(image_bytes))
    # Переводим в Grayscale (L)
    image = image.convert('L')
    # Ресайзим до 28x28 (как в MNIST)
    image = image.resize((28, 28))
    # Превращаем в тензор
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Добавляем размерность batch (1, 1, 28, 28)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем картинку из JSON
        data = request.get_json(force=True)
        image_data = data['image'].split(',')[1]  # Убираем заголовок data:image/png;base64...
        image_bytes = base64.b64decode(image_data)

        tensor = transform_image(image_bytes)

        # Предсказываем
        with torch.no_grad():
            # Logistic
            out_log = model_logistic(tensor)
            _, pred_log = torch.max(out_log, 1)

            # MLP
            out_mlp = model_mlp(tensor)
            _, pred_mlp = torch.max(out_mlp, 1)

            # CNN
            out_cnn = model_cnn(tensor)
            _, pred_cnn = torch.max(out_cnn, 1)

        return jsonify({
            'logistic': int(pred_log.item()),
            'mlp': int(pred_mlp.item()),
            'cnn': int(pred_cnn.item())
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
