import React, { useState } from 'react';
import Papa from 'papaparse';
import './App.css';

function App() {
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [featureCol, setFeatureCol] = useState('');
  const [labelCol, setLabelCol] = useState('');
  const [trainMessage, setTrainMessage] = useState('');
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [completedImage, setCompletedImage] = useState('');

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        setCsvData(results.data.filter(row => Object.values(row).some(val => val !== null && val !== "")));
        setColumns(Object.keys(results.data[0]));
      },
      error: (error) => {
        console.error('CSV parse error:', error);
      }
    });
  };

  const sendTrainingData = async () => {
    if (!featureCol || !labelCol) {
      setTrainMessage('Please select both a feature and a label column.');
      return;
    }

    const features = csvData.map(row => [row[featureCol]]);
    const labels = csvData.map(row => row[labelCol]);

    const payload = { features, labels };

    try {
      const response = await fetch('http://localhost:8080/train_linear', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const json = await response.json();
      setTrainMessage(json.message || 'Training complete');
      setImages(json.training_images || []);
      setCompletedImage(json.completed_image || '');
      setCurrentIndex(0);

    } catch (error) {
      console.error('Error sending training data:', error);
      setTrainMessage('Failed to send training data');
    }
  };

  const nextImage = () => {
    setCurrentIndex(prev => (prev + 1 < images.length ? prev + 1 : 0));
  };

  const prevImage = () => {
    setCurrentIndex(prev => (prev - 1 >= 0 ? prev - 1 : images.length - 1));
  };

  return (
      <div className="App">
        <header className="App-header">
          <h1>MLVisualizer</h1>

          {/* CSV Upload */}
          <input type="file" accept=".csv" onChange={handleFileUpload} />

          {/* Column selectors */}
          {columns.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <label>
                  Feature:
                  <select value={featureCol} onChange={e => setFeatureCol(e.target.value)}>
                    <option value="">--Select Feature--</option>
                    {columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </label>
                <label style={{ marginLeft: '1rem' }}>
                  Label:
                  <select value={labelCol} onChange={e => setLabelCol(e.target.value)}>
                    <option value="">--Select Label--</option>
                    {columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </label>
              </div>
          )}

          {/* Train button */}
          <button onClick={sendTrainingData} disabled={!featureCol || !labelCol}>
            Start Training
          </button>

          {trainMessage && <p>{trainMessage}</p>}

          {/* Image output */}
          {images.length > 0 && (
              <div>
                <h3>Training Progress</h3>
                <img
                    src={`data:image/png;base64,${images[currentIndex]}`}
                    alt={`Epoch ${currentIndex}`}
                    style={{ maxWidth: '500px', borderRadius: '8px', margin: '20px' }}
                />
                <div style={{ marginTop: '1rem' }}>
                  <input
                      type="range"
                      min="0"
                      max={images.length - 1}
                      value={currentIndex}
                      onChange={(e) => setCurrentIndex(Number(e.target.value))}
                      style={{ width: '100%', maxWidth: '500px' }}
                  />
                  <div style={{ marginTop: '0.5rem' }}>
                    Epoch {currentIndex + 1} of {images.length}
                  </div>
                </div>

              </div>
          )}

          {completedImage && (
              <div>
                <h3>Final Model Output</h3>
                <img
                    src={`data:image/png;base64,${completedImage}`}
                    alt="Final Output"
                    style={{ maxWidth: '500px', borderRadius: '8px', margin: '20px' }}
                />
              </div>
          )}

        </header>
      </div>
  );
}

export default App;
