import React, { useState } from 'react';
import './App.css';

function App() {
  const [trainMessage, setTrainMessage] = useState('');
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [modelType, setModelType] = useState('');

  // Test data for training
  const testPayload = {
    features: [
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 1.0],
      [1.5, 3.0],
      [3.5, 2.5],
      [4.5, 5.0]
    ],
    labels: [2.0, 4.0, 6.0, 3.0, 5.0, 7.0]  // Example continuous values for regression
  };


  // Send training data to the selected model endpoint
  const sendTrainingData = async () => {
    if (!modelType) {
      setTrainMessage('Please select a model type first.');
      return;
    }

    const url =
        modelType === 'logistic'
            ? 'http://localhost:8080/train_logistic'
            : 'http://localhost:8080/train_linear';

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(testPayload)
      });

      const json = await response.json();
      setTrainMessage(json.message);
      setImages(json.images || []);
      setCurrentIndex(0);
    } catch (error) {
      console.error('Error sending training data:', error);
      setTrainMessage('Failed to send training data');
    }
  };

  const handleModelSelection = (type) => {
    setModelType(type);
  };

  const nextImage = () => {
    setCurrentIndex((prevIndex) =>
        prevIndex + 1 < images.length ? prevIndex + 1 : 0
    );
  };

  const prevImage = () => {
    setCurrentIndex((prevIndex) =>
        prevIndex - 1 >= 0 ? prevIndex - 1 : images.length - 1
    );
  };

  return (
      <div className="App">
        <header className="App-header">
          <h1>MLVisualizer</h1>

          {/* Buttons for selecting model type */}
          <button onClick={() => handleModelSelection('logistic')}>Train Logistic Model</button>
          <button onClick={() => handleModelSelection('linear')}>Train Linear Model</button>

          {/* Button to start training */}
          <button onClick={sendTrainingData} disabled={!modelType}>
            Start Training
          </button>

          {/* Display message */}
          {trainMessage && <p>{trainMessage}</p>}

          {/* Display training images (epoch progress) */}
          {images.length > 0 && (
              <div>
                <img
                    src={`data:image/png;base64,${images[currentIndex]}`}
                    alt={`Epoch ${currentIndex}`}
                    style={{ maxWidth: '500px', borderRadius: '8px', margin: '20px' }}
                />
                <div>
                  <button onClick={prevImage}>◀ Prev</button>
                  <span style={{ margin: '0 10px' }}>Epoch {currentIndex}</span>
                  <button onClick={nextImage}>Next ▶</button>
                </div>
              </div>
          )}
        </header>
      </div>
  );
}

export default App;
