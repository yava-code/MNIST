# MNIST Digit Recognition Project

A PyTorch-based web application for comparing different neural network architectures on the MNIST handwritten digit dataset.

## 🎯 Project Overview

This project implements and compares three different neural network architectures for digit recognition:
- **Logistic Regression** - Simple linear classifier (baseline)
- **MLP (Multi-Layer Perceptron)** - Fully connected neural network
- **CNN (Convolutional Neural Network)** - Deep learning with convolutional layers

The web interface lets you draw digits and see real-time predictions from all three models!

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yava-code/MNIST.git
cd MNIST
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask app:
```bash
python main.py
```

4. Open your browser and go to `http://localhost:5000`

## 📊 Model Performance

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Logistic Regression | Single layer | 86.2% |
| MLP | 784→128→64→10 | 41.4% |
| CNN | Conv layers + MaxPool | 55.3% |

*Note: MLP and CNN models were stopped early and need more training epochs for better convergence.*

## 🏗️ Project Structure

```
MNIST/
├── models/              # Trained model weights (.pth files)
├── templates/           # HTML templates for Flask
├── tests/              # Unit tests
├── main.py             # Flask application
├── main.ipynb          # Training notebook
├── index.html          # Project overview page
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🐳 Docker

Build and run with Docker:
```bash
docker build -t mnist-app .
docker run -p 5000:5000 mnist-app
```

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **Flask** - Web framework
- **Vanilla JavaScript** - Frontend interaction
- **HTML5 Canvas** - Drawing interface

## 📝 How It Works

1. **Training**: Models are trained on the MNIST dataset (60k training images)
2. **Saving**: Model weights are saved as `.pth` files
3. **Loading**: Flask app loads all three models at startup
4. **Inference**: User draws a digit → converted to 28x28 tensor → predictions from all models

## 🔮 Future Improvements

- [ ] Train models for more epochs
- [ ] Add model confidence scores
- [ ] Implement data augmentation
- [ ] Add more architectures (ResNet, Transformer)
- [ ] Deploy to cloud (Heroku/Railway)

## 👨‍💻 Author

Created as a learning project to understand different neural network architectures and model deployment.

## 📄 License

MIT License - feel free to use this for your own learning!
