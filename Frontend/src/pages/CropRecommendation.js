import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import "./Prediction.css";

const API_BASE = "http://localhost:8000";

function CropRecommendation() {
  const [formData, setFormData] = useState({
    State_Name: "kerala",
    Season: "kharif",
    N: 90,
    P: 42,
    K: 43,
    pH: 6.5,
    rainfall: 202,
    temperature: 23,
  });
  const [filterOptions, setFilterOptions] = useState({
    states: [],
    seasons: [],
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
      const response = await axios.post(`${API_BASE}/recommend/crop`, {
        samples: [formData],
      });
      setResult(response.data.recommendations[0]);
    } catch (err) {
      setError(
        "Recommendation failed: " + (err.response?.data?.detail || err.message)
      );
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (example) => {
    const examples = {
      rice: {
        State_Name: "west bengal",
        Season: "kharif",
        N: 80,
        P: 40,
        K: 40,
        pH: 6.5,
        rainfall: 220,
        temperature: 25,
      },
      wheat: {
        State_Name: "punjab",
        Season: "rabi",
        N: 100,
        P: 50,
        K: 45,
        pH: 7.0,
        rainfall: 180,
        temperature: 20,
      },
    };
    setFormData(examples[example]);
  };

  return (
    <div className="prediction-page">
      <div className="page-header">
        <h1>Crop Recommendation</h1>
        <p>
          Get top 3 crop recommendations using Custom k-NN Classifier
          (distance-weighted voting)
        </p>
      </div>

      <div className="prediction-layout">
        <div className="card input-card">
          <h3>Input Features</h3>

          <div className="example-buttons">
            <button
              className="btn btn-secondary"
              onClick={() => loadExample("rice")}
            >
              Load Rice-Suitable
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => loadExample("wheat")}
            >
              Load Wheat-Suitable
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
            {loading ? "Recommending..." : "Get Recommendations"}
          </button>
        </div>

        <div className="card result-card">
          <h3>Recommended Crops</h3>

          {error && <div className="error">{error}</div>}

          {result && (
            <div className="result-display">
              <div className="result-label">Top 3 Crop Recommendations</div>

              {/* Confidence Scores Chart */}
              {result.confidence_scores && (
                <div className="conditions-viz" style={{ marginTop: "1.5rem" }}>
                  <h4>Recommendation Confidence Scores</h4>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart
                      data={result.top_3_crops.map((crop, idx) => ({
                        crop: crop,
                        confidence: result.confidence_scores[idx] || 0,
                      }))}
                      layout="horizontal"
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="category"
                        dataKey="crop"
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis
                        type="number"
                        label={{
                          value: "Confidence %",
                          angle: -90,
                          position: "insideLeft",
                        }}
                        domain={[0, 100]}
                      />
                      <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                      <Bar dataKey="confidence" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                  <p
                    style={{
                      textAlign: "center",
                      fontSize: "0.875rem",
                      color: "#6b7280",
                      marginTop: "0.5rem",
                    }}
                  >
                    Higher confidence indicates stronger recommendation based on
                    similar conditions
                  </p>
                </div>
              )}

              <div className="recommendations-list">
                {result.top_3_crops.map((crop, idx) => (
                  <div key={idx} className="recommendation-item">
                    <span className="rank">#{idx + 1}</span>
                    <span className="crop-name">{crop}</span>
                    {result.confidence_scores && (
                      <span className="confidence-badge">
                        {result.confidence_scores[idx]?.toFixed(1)}%
                      </span>
                    )}
                  </div>
                ))}
              </div>

              {/* Soil Nutrients Visualization */}
              <div className="conditions-viz">
                <h4>Soil Nutrient Profile (NPK)</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={[
                      { nutrient: "Nitrogen (N)", value: formData.N },
                      { nutrient: "Phosphorus (P)", value: formData.P },
                      { nutrient: "Potassium (K)", value: formData.K },
                    ]}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="nutrient" />
                    <YAxis
                      label={{
                        value: "mg/kg",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Bar dataKey="value" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Environmental Conditions Radar */}
              {/* <div className="conditions-viz">
                <h4>🌍 Environmental Conditions Profile</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart
                    data={[
                      {
                        condition: `N: ${formData.N}`,
                        value: (formData.N / 150) * 100,
                        fullMark: 100,
                      },
                      {
                        condition: `P: ${formData.P}`,
                        value: (formData.P / 150) * 100,
                        fullMark: 100,
                      },
                      {
                        condition: `K: ${formData.K}`,
                        value: (formData.K / 150) * 100,
                        fullMark: 100,
                      },
                      {
                        condition: `pH: ${formData.pH}`,
                        value: (formData.pH / 14) * 100,
                        fullMark: 100,
                      },
                      {
                        condition: `Rain: ${formData.rainfall}mm`,
                        value: (formData.rainfall / 400) * 100,
                        fullMark: 100,
                      },
                      {
                        condition: `Temp: ${formData.temperature}°C`,
                        value: (formData.temperature / 50) * 100,
                        fullMark: 100,
                      },
                    ]}
                  >
                    <PolarGrid />
                    <PolarAngleAxis
                      dataKey="condition"
                      tick={{ fontSize: 11 }}
                    />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Conditions"
                      dataKey="value"
                      stroke="#4f46e5"
                      fill="#4f46e5"
                      fillOpacity={0.6}
                    />
                    <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                  </RadarChart>
                </ResponsiveContainer>
                <p
                  style={{
                    textAlign: "center",
                    fontSize: "0.875rem",
                    color: "#6b7280",
                    marginTop: "0.5rem",
                  }}
                >
                  Normalized values showing relative levels of each parameter
                </p>
              </div> */}

              <div className="result-info">
                <p>
                  <strong>Location:</strong>{" "}
                  {formData.State_Name.charAt(0).toUpperCase() +
                    formData.State_Name.slice(1)}{" "}
                  •{" "}
                  {formData.Season.charAt(0).toUpperCase() +
                    formData.Season.slice(1)}{" "}
                  Season
                </p>
                <p>
                  <strong>Algorithm:</strong> Custom k-NN Classifier
                </p>
                <p>
                  <strong>Method:</strong> Distance-weighted voting (k=15)
                </p>
                <p>
                  <strong>Distance Metric:</strong> Minkowski (p=2, Euclidean)
                </p>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="placeholder">
              <p>Enter values and click "Get Recommendations" to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CropRecommendation;
