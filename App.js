import React, { useState } from 'react';

function App() {
  const [ticker, setTicker] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Prediction failed');
      }
      const data = await response.json();
      setPrediction(data.latest_prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: '40px auto', padding: 24, border: '1px solid #eee', borderRadius: 8 }}>
      <h2>Stock Price Prediction</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor="ticker">Stock Ticker:</label>
        <input
          id="ticker"
          type="text"
          value={ticker}
          onChange={e => setTicker(e.target.value)}
          style={{ marginLeft: 8, marginRight: 8 }}
          required
        />
        <button type="submit" disabled={loading || !ticker.trim()}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>
      {error && <div style={{ color: 'red', marginTop: 16 }}>{error}</div>}
      {prediction !== null && !error && (
        <div style={{ marginTop: 24 }}>
          <strong>Predicted Close Price:</strong> ${prediction.toFixed(2)}
        </div>
      )}
    </div>
  );
}

export default App;
