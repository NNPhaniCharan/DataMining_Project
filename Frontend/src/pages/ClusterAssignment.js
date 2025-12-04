import React, { useState, useEffect } from "react";
import axios from "axios";
import "./Prediction.css";

const API_BASE = "http://localhost:8000";

// Cluster labels based on analysis of cluster characteristics
const CLUSTER_LABELS = {
  0: {
    name: "Spice & Oilseed Zone",
    description: "Warm, low-nutrient soils for sesamum, coriander, garlic",
    color: "#FF9800",
  },
  1: {
    name: "Vegetable & Fruit Zone",
    description: "Rich soils for onion, banana, tomato",
    color: "#4CAF50",
  },
  2: {
    name: "Plantation Spice Zone",
    description: "Acidic, wet soils for turmeric, cardamom",
    color: "#9C27B0",
  },
  3: {
    name: "Dry Cereal Zone",
    description: "Low rainfall for rapeseed, rice, barley, maize",
    color: "#795548",
  },
  4: {
    name: "Wheat Belt",
    description: "Cool, dry conditions ideal for wheat",
    color: "#FFC107",
  },
  5: {
    name: "Wet Tropical Zone",
    description: "High rainfall for rice, maize, tapioca",
    color: "#00BCD4",
  },
  6: {
    name: "Hot Staple Zone",
    description: "Hot climate for rice, maize, cotton",
    color: "#F44336",
  },
  7: {
    name: "Potato Zone",
    description: "High nitrogen soils for intensive potato farming",
    color: "#8BC34A",
  },
  8: {
    name: "Root Crop Zone",
    description: "High potassium for sweet potato, arecanut",
    color: "#E91E63",
  },
};

function ClusterAssignment() {
  const [formData, setFormData] = useState({
    N: 90,
    P: 42,
    K: 43,
    pH: 6.5,
    rainfall: 202,
    temperature: 23,
  });
  const [result, setResult] = useState(null);
  const [clusterInfo, setClusterInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchClusterInfo();
  }, []);

  const fetchClusterInfo = async () => {
    try {
      const res = await axios.get(`${API_BASE}/evaluation/cluster/metrics`);
      setClusterInfo(res.data);
    } catch (err) {
      console.error("Failed to fetch cluster info:", err);
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
      const response = await axios.post(`${API_BASE}/cluster/assign`, {
        samples: [formData],
      });
      setResult(response.data.assignments[0]);
    } catch (err) {
      setError(
        "Cluster assignment failed: " +
          (err.response?.data?.detail || err.message)
      );
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (clusterId) => {
    if (clusterInfo && clusterInfo.cluster_centers[clusterId]) {
      const center = clusterInfo.cluster_centers[clusterId];
      setFormData({
        N: Math.round(center.N),
        P: Math.round(center.P),
        K: Math.round(center.K),
        pH: parseFloat(center.pH.toFixed(1)),
        rainfall: Math.round(center.rainfall),
        temperature: parseFloat(center.temperature.toFixed(1)),
      });
    }
  };

  const getClusterDetails = (clusterId) => {
    if (!clusterInfo) return null;
    return clusterInfo.cluster_centers.find((c) => c.cluster_id === clusterId);
  };

  return (
    <div className="prediction-page">
      <div className="page-header">
        <h1>Cluster Assignment</h1>
        <p>
          Assign soil/climate conditions to agro-regime clusters using Custom
          K-Means (k-means++ init)
        </p>
      </div>

      <div className="prediction-layout">
        <div className="card input-card">
          <h3>Input Features</h3>

          {clusterInfo && (
            <div className="example-buttons cluster-examples">
              <p style={{ fontSize: "0.875rem", marginBottom: "0.5rem" }}>
                Load cluster center examples:
              </p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                {clusterInfo.cluster_centers.map((cluster) => (
                  <button
                    key={cluster.cluster_id}
                    className="btn btn-secondary"
                    onClick={() => loadExample(cluster.cluster_id)}
                    title={CLUSTER_LABELS[cluster.cluster_id]?.name}
                    style={{
                      borderLeft: `3px solid ${
                        CLUSTER_LABELS[cluster.cluster_id]?.color || "#666"
                      }`,
                    }}
                  >
                    {cluster.cluster_id}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="form-grid">
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
            {loading ? "Assigning..." : "Assign Cluster"}
          </button>
        </div>

        <div className="card result-card">
          <h3>Cluster Assignment</h3>

          {error && <div className="error">{error}</div>}

          {result && (
            <div className="result-display">
              <div className="result-label">Assigned Cluster</div>
              <div className="result-value">Cluster {result.cluster_id}</div>

              {/* Cluster Label & Description */}
              {CLUSTER_LABELS[result.cluster_id] && (
                <div
                  style={{
                    marginTop: "1rem",
                    padding: "1rem",
                    background: `${CLUSTER_LABELS[result.cluster_id].color}15`,
                    border: `2px solid ${
                      CLUSTER_LABELS[result.cluster_id].color
                    }`,
                    borderRadius: "12px",
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{
                      fontSize: "1.25rem",
                      fontWeight: "700",
                      color: CLUSTER_LABELS[result.cluster_id].color,
                      marginBottom: "0.5rem",
                    }}
                  >
                    {CLUSTER_LABELS[result.cluster_id].name}
                  </div>
                  <div
                    style={{
                      fontSize: "0.9rem",
                      color: "#555",
                      lineHeight: "1.4",
                    }}
                  >
                    {CLUSTER_LABELS[result.cluster_id].description}
                  </div>
                </div>
              )}

              {getClusterDetails(result.cluster_id) && (
                <div className="cluster-details">
                  <h4>Cluster Center Values:</h4>
                  <div className="cluster-stats">
                    <div>
                      <strong>N:</strong>{" "}
                      {getClusterDetails(result.cluster_id).N.toFixed(2)}
                    </div>
                    <div>
                      <strong>P:</strong>{" "}
                      {getClusterDetails(result.cluster_id).P.toFixed(2)}
                    </div>
                    <div>
                      <strong>K:</strong>{" "}
                      {getClusterDetails(result.cluster_id).K.toFixed(2)}
                    </div>
                    <div>
                      <strong>pH:</strong>{" "}
                      {getClusterDetails(result.cluster_id).pH.toFixed(2)}
                    </div>
                    <div>
                      <strong>Rainfall:</strong>{" "}
                      {getClusterDetails(result.cluster_id).rainfall.toFixed(2)}{" "}
                      mm
                    </div>
                    <div>
                      <strong>Temperature:</strong>{" "}
                      {getClusterDetails(result.cluster_id).temperature.toFixed(
                        2
                      )}{" "}
                      °C
                    </div>
                    <div>
                      <strong>Samples:</strong>{" "}
                      {getClusterDetails(
                        result.cluster_id
                      ).count.toLocaleString()}
                    </div>
                  </div>

                  {/* Yield Profile Section */}
                  {getClusterDetails(result.cluster_id).avg_yield !==
                    undefined && (
                    <div style={{ marginTop: "1.5rem" }}>
                      <h4>Typical Yield Profile:</h4>
                      <div
                        className="cluster-stats"
                        style={{
                          background: "#f8f9fa",
                          padding: "1rem",
                          borderRadius: "8px",
                          marginTop: "0.5rem",
                        }}
                      >
                        <div>
                          <strong>Average Yield:</strong>{" "}
                          <span style={{ color: "#28a745", fontWeight: "600" }}>
                            {getClusterDetails(
                              result.cluster_id
                            ).avg_yield.toFixed(2)}{" "}
                            t/ha
                          </span>
                        </div>
                        <div>
                          <strong>Median Yield:</strong>{" "}
                          {getClusterDetails(
                            result.cluster_id
                          ).median_yield.toFixed(2)}{" "}
                          t/ha
                        </div>
                        <div>
                          <strong>Yield Range:</strong>{" "}
                          {getClusterDetails(
                            result.cluster_id
                          ).min_yield.toFixed(2)}{" "}
                          -{" "}
                          {getClusterDetails(
                            result.cluster_id
                          ).max_yield.toFixed(2)}{" "}
                          t/ha
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Dominant Crops Section */}
                  {getClusterDetails(result.cluster_id).top_crops &&
                    getClusterDetails(result.cluster_id).top_crops.length >
                      0 && (
                      <div style={{ marginTop: "1.5rem" }}>
                        <h4>Dominant Crops:</h4>
                        <div style={{ marginTop: "0.5rem" }}>
                          {getClusterDetails(result.cluster_id).top_crops.map(
                            (cropInfo, idx) => (
                              <div
                                key={idx}
                                style={{
                                  background: idx === 0 ? "#e8f5e9" : "#f8f9fa",
                                  padding: "0.75rem",
                                  marginBottom: "0.5rem",
                                  borderRadius: "6px",
                                  border:
                                    idx === 0
                                      ? "2px solid #4caf50"
                                      : "1px solid #e0e0e0",
                                }}
                              >
                                <div
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    marginBottom: "0.25rem",
                                  }}
                                >
                                  <strong
                                    style={{
                                      textTransform: "capitalize",
                                      color: idx === 0 ? "#2e7d32" : "#333",
                                    }}
                                  >
                                    {idx + 1}. {cropInfo.crop}
                                  </strong>
                                  <span
                                    style={{
                                      fontSize: "0.875rem",
                                      color: "#666",
                                      background: "white",
                                      padding: "0.25rem 0.5rem",
                                      borderRadius: "12px",
                                    }}
                                  >
                                    {cropInfo.percentage.toFixed(1)}%
                                  </span>
                                </div>
                                <div
                                  style={{
                                    fontSize: "0.875rem",
                                    color: "#666",
                                    display: "flex",
                                    justifyContent: "space-between",
                                  }}
                                >
                                  <span>
                                    {cropInfo.count.toLocaleString()} samples
                                  </span>
                                  <span
                                    style={{
                                      color: "#28a745",
                                      fontWeight: "500",
                                    }}
                                  >
                                    Avg: {cropInfo.avg_yield.toFixed(2)} t/ha
                                  </span>
                                </div>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    )}
                </div>
              )}

              <div className="result-info">
                <p>
                  <strong>Algorithm:</strong> Custom K-Means Clustering
                </p>
                <p>
                  <strong>Initialization:</strong> k-means++
                </p>
                <p>
                  <strong>Total Clusters:</strong>{" "}
                  {clusterInfo?.metrics?.n_clusters}
                </p>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="placeholder">
              <p>Enter values and click "Assign Cluster" to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ClusterAssignment;
