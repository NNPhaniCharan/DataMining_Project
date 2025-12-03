import React, { useState, useEffect } from "react";
import axios from "axios";
import "./Prediction.css";

const API_BASE = "http://localhost:8000";

function YieldPrediction() {
  const [formData, setFormData] = useState({
    State_Name: "kerala",
    Season: "kharif",
    Crop: "rice",
    N: 90,
    P: 42,
    K: 43,
    pH: 6.5,
    rainfall: 650,
    temperature: 23,
  });
  const [filterOptions, setFilterOptions] = useState({
    states: [],
    seasons: [],
    crops: [],
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFilterOptions();
  }, []);

  const fetchFilterOptions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/data/filters`);
      setFilterOptions(res.data);
    } catch (err) {
      console.error("Error fetching filter options:", err);
    }
  };

  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/predict/yield`, {
        samples: [formData],
      });
      setResult(response.data.predictions[0]);
    } catch (err) {
      setError(
        "Prediction failed: " + (err.response?.data?.detail || err.message)
      );
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (example) => {
    const examples = {
      highYield: {
        State_Name: "punjab",
        Season: "rabi",
        Crop: "wheat",
        N: 120,
        P: 60,
        K: 50,
        pH: 7.0,
        rainfall: 800,
        temperature: 22,
      },
      lowYield: {
        State_Name: "rajasthan",
        Season: "kharif",
        Crop: "bajra",
        N: 40,
        P: 20,
        K: 20,
        pH: 7.5,
        rainfall: 350,
        temperature: 28,
      },
    };
    setFormData(examples[example]);
  };

  // Helper function to get yield category and color
  const getYieldCategory = (yieldValue) => {
    if (yieldValue < 2)
      return { category: "Low", color: "#ff6b6b", percent: 33 };
    if (yieldValue < 4)
      return { category: "Medium", color: "#ffd93d", percent: 66 };
    return { category: "High", color: "#6bcf7f", percent: 100 };
  };

  return (
    <div className="prediction-page">
      <div className="page-header">
        <h1>🌾 Yield Prediction</h1>
        <p>
          Predict crop yield using Custom Gradient Boosting (from scratch,
          decision trees with gradient descent)
        </p>
      </div>

      <div className="prediction-layout">
        <div className="card input-card">
          <h3>Input Features</h3>

          <div className="example-buttons">
            <button
              className="btn btn-secondary"
              onClick={() => loadExample("highYield")}
            >
              Load High-Yield Example
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => loadExample("lowYield")}
            >
              Load Low-Yield Example
            </button>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label>State Name</label>
              <select
                value={formData.State_Name}
                onChange={(e) => handleChange("State_Name", e.target.value)}
              >
                {filterOptions.states?.map((state) => (
                  <option key={state} value={state}>
                    {state}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Season</label>
              <select
                value={formData.Season}
                onChange={(e) => handleChange("Season", e.target.value)}
              >
                {filterOptions.seasons?.map((season) => (
                  <option key={season} value={season}>
                    {season}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Crop</label>
              <select
                value={formData.Crop}
                onChange={(e) => handleChange("Crop", e.target.value)}
              >
                {filterOptions.crops?.map((crop) => (
                  <option key={crop} value={crop}>
                    {crop}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Nitrogen (N)</label>
              <input
                type="number"
                value={formData.N}
                onChange={(e) => handleChange("N", parseFloat(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label>Phosphorus (P)</label>
              <input
                type="number"
                value={formData.P}
                onChange={(e) => handleChange("P", parseFloat(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label>Potassium (K)</label>
              <input
                type="number"
                value={formData.K}
                onChange={(e) => handleChange("K", parseFloat(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label>pH Level</label>
              <input
                type="number"
                step="0.1"
                value={formData.pH}
                onChange={(e) => handleChange("pH", parseFloat(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label>Rainfall (mm)</label>
              <input
                type="number"
                value={formData.rainfall}
                onChange={(e) =>
                  handleChange("rainfall", parseFloat(e.target.value))
                }
              />
            </div>

            <div className="form-group">
              <label>Temperature (°C)</label>
              <input
                type="number"
                step="0.1"
                value={formData.temperature}
                onChange={(e) =>
                  handleChange("temperature", parseFloat(e.target.value))
                }
              />
            </div>
          </div>

          <button
            className="btn btn-primary predict-btn"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "Predicting..." : "🔮 Predict Yield"}
          </button>
        </div>

        <div className="card result-card">
          <h3>Analysis & Results</h3>

          {error && <div className="error">{error}</div>}

          {result && (
            <div className="result-display">
              {/* Key Result Summary */}
              <div style={{ marginBottom: "2rem", textAlign: "center" }}>
                <div className="result-label">Predicted Yield</div>
                <div
                  className="result-value"
                  style={{
                    color: getYieldCategory(result.predicted_yield_ton_per_hec)
                      .color,
                  }}
                >
                  {result.predicted_yield_ton_per_hec.toFixed(3)}{" "}
                  <span>tons/hectare</span>
                </div>

                <div
                  style={{
                    display: "inline-block",
                    padding: "0.5rem 1rem",
                    background:
                      getYieldCategory(result.predicted_yield_ton_per_hec)
                        .color + "20",
                    color: getYieldCategory(result.predicted_yield_ton_per_hec)
                      .color,
                    borderRadius: "2rem",
                    fontWeight: "600",
                    marginBottom: "1rem",
                  }}
                >
                  {
                    getYieldCategory(result.predicted_yield_ton_per_hec)
                      .category
                  }{" "}
                  Yield
                </div>
              </div>

              {/* Model Info */}
              <div
                style={{
                  marginTop: "2rem",
                  padding: "1rem",
                  background: "#f8f9fa",
                  borderRadius: "0.5rem",
                  fontSize: "0.8rem",
                }}
              >
                <p style={{ marginBottom: "0.25rem" }}>
                  <strong>Algorithm:</strong> Custom Gradient Boosting
                </p>
                <p style={{ marginBottom: "0.25rem" }}>
                  <strong>Method:</strong> Sequential ensemble of decision trees
                  trained via gradient descent
                </p>
                <p style={{ margin: 0 }}>
                  <strong>Approach:</strong> Minimizes loss by iteratively
                  fitting trees to negative gradients
                </p>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="placeholder">
              <p>Enter values and click "Predict Yield" to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default YieldPrediction;
