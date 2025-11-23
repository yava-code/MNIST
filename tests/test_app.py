import pytest
import json
import base64
import sys
from io import BytesIO
from PIL import Image
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_test_image():
    """Create a simple test image (black square with white center)"""
    img = Image.new('L', (28, 28), color=0)  # Black background
    # Draw a white square in the center (simulate a digit)
    for i in range(10, 18):
        for j in range(10, 18):
            img.putpixel((i, j), 255)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return f"data:image/png;base64,{img_base64}"


class TestFlaskApp:
    """Test Flask application endpoints"""
    
    def test_home_page(self, client):
        """Test that home page loads"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'MNIST' in response.data or b'mnist' in response.data
    
    def test_predict_endpoint_exists(self, client):
        """Test that predict endpoint exists"""
        # Send empty POST to check endpoint exists
        response = client.post('/predict', 
                               data=json.dumps({}),
                               content_type='application/json')
        # Should return error or process, but not 404
        assert response.status_code != 404
    
    def test_predict_with_image(self, client):
        """Test prediction with a valid image"""
        test_image = create_test_image()
        response = client.post('/predict',
                              data=json.dumps({'image': test_image}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check that all three model predictions are present
        assert 'logistic' in data
        assert 'mlp' in data
        assert 'cnn' in data
        
        # Check that predictions are valid digits (0-9)
        assert 0 <= data['logistic'] <= 9
        assert 0 <= data['mlp'] <= 9
        assert 0 <= data['cnn'] <= 9
    
    def test_predict_returns_integers(self, client):
        """Test that predictions are integers"""
        test_image = create_test_image()
        response = client.post('/predict',
                              data=json.dumps({'image': test_image}),
                              content_type='application/json')
        
        data = json.loads(response.data)
        assert isinstance(data['logistic'], int)
        assert isinstance(data['mlp'], int)
        assert isinstance(data['cnn'], int)
    
    def test_invalid_image_data(self, client):
        """Test handling of invalid image data"""
        response = client.post('/predict',
                              data=json.dumps({'image': 'invalid_data'}),
                              content_type='application/json')
        
        data = json.loads(response.data)
        # Should return error
        assert 'error' in data or response.status_code != 200
