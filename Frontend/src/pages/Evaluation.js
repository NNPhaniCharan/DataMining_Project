import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./Evaluation.css";

const API_BASE = "http://localhost:8000";
const COLORS = [
  "#4f46e5",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#14b8a6",
  "#f97316",
];

function Evaluation() {
  const [yieldMetrics, setYieldMetrics] = useState(null);
  const [cropMetrics, setCropMetrics] = useState(null);
  const [clusterMetrics, setClusterMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Helper function to format numbers and avoid "-0.00" display
  const formatNumber = (value, decimals = 4) => {
    if (value === null || value === undefined) return "N/A";
    const formatted = value.toFixed(decimals);
    // If the value is -0.0000, display as 0.0000
    if (parseFloat(formatted) === 0) {
      return (0).toFixed(decimals);
    }
    return formatted;
  };

  useEffect(() => {
    fetchAllMetrics();
  }, []);

  const fetchAllMetrics = async () => {
    setLoading(true);
    try {
      const [yieldRes, cropRes, clusterRes] = await Promise.all([
        axios.get(`${API_BASE}/evaluation/yield/metrics`),
        axios.get(`${API_BASE}/evaluation/croprec/metrics`),
        axios.get(`${API_BASE}/evaluation/cluster/metrics`),
      ]);

      setYieldMetrics(yieldRes.data);
      setCropMetrics(cropRes.data);
      setClusterMetrics(clusterRes.data);
    } catch (err) {
      setError("Failed to load evaluation metrics: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading)
    return <div className="loading">Loading evaluation metrics...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="evaluation-page">
      <div className="page-header">
        <h1>📈 Model Evaluation Dashboard</h1>
        <p>
          Comprehensive metrics, visualizations, and key outputs for all custom
          algorithms
        </p>
      </div>

      {/* Yield Prediction Model */}
      <div className="card">
        <h2>🌾 Yield Prediction - Custom Gradient Boosting</h2>
        <p className="algo-desc">{yieldMetrics?.metrics?.algorithm}</p>

        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">R² Score</div>
            <div className="metric-value">
              {yieldMetrics?.metrics?.r2?.toFixed(4)}
            </div>
            <div className="metric-subtitle">Coefficient of Determination</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">RMSE</div>
            <div className="metric-value">
              {yieldMetrics?.metrics?.rmse?.toFixed(4)}
            </div>
            <div className="metric-subtitle">Root Mean Squared Error</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">MAE</div>
            <div className="metric-value">
              {yieldMetrics?.metrics?.mae?.toFixed(4)}
            </div>
            <div className="metric-subtitle">Mean Absolute Error</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Best n_estimators</div>
            <div className="metric-value">
              {yieldMetrics?.metrics?.best_n_estimators}
            </div>
            <div className="metric-subtitle">Optimal Trees (CV)</div>
          </div>
        </div>

        <div className="section-header">
          <h3>Top Feature Importances</h3>
          <p className="train-info">
            Trained on {yieldMetrics?.metrics?.n_train?.toLocaleString()}{" "}
            samples, tested on {yieldMetrics?.metrics?.n_test?.toLocaleString()}{" "}
            samples
          </p>
        </div>

        <div className="coefficients-section">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={yieldMetrics?.feature_importances?.slice(0, 15)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="feature"
                angle={-45}
                textAnchor="end"
                height={120}
                fontSize={12}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar
                dataKey="importance"
                fill="#4f46e5"
                name="Feature Importance"
              />
            </BarChart>
          </ResponsiveContainer>

          <div className="coef-table-wrapper">
            <table className="coef-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Feature</th>
                  <th>Importance</th>
                  <th>Importance %</th>
                </tr>
              </thead>
              <tbody>
                {yieldMetrics?.feature_importances
                  ?.slice(0, 20)
                  .map((item, idx) => (
                    <tr key={idx}>
                      <td>{idx + 1}</td>
                      <td className="feature-name">{item.feature}</td>
                      <td className="positive">{item.importance.toFixed(6)}</td>
                      <td>{(item.importance * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          <div className="intercept-info">
            <strong>Learning Rate:</strong>{" "}
            {yieldMetrics?.metrics?.learning_rate} |<strong> Max Depth:</strong>{" "}
            {yieldMetrics?.metrics?.max_depth} |
            <strong> Total Features:</strong> {yieldMetrics?.total_features}
          </div>
        </div>
      </div>

      {/* Crop Recommendation Model */}
      <div className="card">
        <h2>🌱 Crop Recommendation - Custom k-NN Classifier</h2>
        <p className="algo-desc">{cropMetrics?.metrics?.algorithm}</p>

        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">Top-1 Accuracy</div>
            <div className="metric-value">
              {(cropMetrics?.metrics?.top1_accuracy * 100)?.toFixed(2)}%
            </div>
            <div className="metric-subtitle">Exact match accuracy</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Top-3 Accuracy</div>
            <div className="metric-value">
              {(cropMetrics?.metrics?.top3_accuracy * 100)?.toFixed(2)}%
            </div>
            <div className="metric-subtitle">Top 3 predictions</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">F1-Score (Macro)</div>
            <div className="metric-value">
              {cropMetrics?.metrics?.f1_macro?.toFixed(4)}
            </div>
            <div className="metric-subtitle">Macro-averaged F1</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">k Neighbors</div>
            <div className="metric-value">
              {cropMetrics?.metrics?.n_neighbors}
            </div>
            <div className="metric-subtitle">Distance-weighted voting</div>
          </div>
        </div>

        <div className="section-header">
          <h3>Class Distribution</h3>
          <p>Total of {cropMetrics?.n_classes} unique crop classes</p>
        </div>

        {cropMetrics?.classes && (
          <div className="classes-list">
            <strong>Classified Crops:</strong>{" "}
            {cropMetrics.classes.slice(0, 20).join(", ")}
            {cropMetrics.classes.length > 20 &&
              ` ... and ${cropMetrics.classes.length - 20} more`}
          </div>
        )}
      </div>

      {/* Clustering Model */}
      <div className="card">
        <h2>🎯 Cluster Assignment - Custom K-Means</h2>
        <p className="algo-desc">{clusterMetrics?.metrics?.algorithm}</p>

        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">Number of Clusters</div>
            <div className="metric-value">
              {clusterMetrics?.metrics?.n_clusters}
            </div>
            <div className="metric-subtitle">k-means++ initialization</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Silhouette Score</div>
            <div className="metric-value">
              {clusterMetrics?.metrics?.silhouette_score?.toFixed(4)}
            </div>
            <div className="metric-subtitle">Cluster cohesion [-1, 1]</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Calinski-Harabasz</div>
            <div className="metric-value">
              {clusterMetrics?.metrics?.calinski_harabasz_score?.toFixed(2)}
            </div>
            <div className="metric-subtitle">
              Variance ratio (higher is better)
            </div>
          </div>
        </div>

        <div className="quality-info">
          <h4>📊 Clustering Quality Metrics</h4>
          <p>
            <strong>Silhouette Score:</strong> Measures how similar an object is
            to its own cluster compared to other clusters. Range: [-1, 1].
            Higher is better. Values near 0 indicate overlapping clusters.
          </p>
          <p>
            <strong>Calinski-Harabasz Index:</strong> Ratio of between-cluster
            dispersion to within-cluster dispersion. Higher values indicate
            better-defined clusters.
          </p>
        </div>

        <div className="section-header">
          <h3>Cluster Sizes</h3>
        </div>

        {clusterMetrics?.metrics?.counts && (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={clusterMetrics.metrics.counts.map((count, idx) => ({
                  name: `Cluster ${idx}`,
                  value: count,
                }))}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name}: ${(percent * 100).toFixed(1)}%`
                }
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {clusterMetrics.metrics.counts.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        )}

        <div className="section-header">
          <h3>Cluster Centers (Original Scale)</h3>
        </div>

        {clusterMetrics?.cluster_centers && (
          <div className="table-responsive">
            <table className="cluster-table">
              <thead>
                <tr>
                  <th>Cluster</th>
                  {clusterMetrics.features.map((feat) => (
                    <th key={feat}>{feat}</th>
                  ))}
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {clusterMetrics.cluster_centers.map((cluster, idx) => (
                  <tr key={idx}>
                    <td>
                      <strong>Cluster {cluster.cluster_id}</strong>
                    </td>
                    {clusterMetrics.features.map((feat) => (
                      <td key={feat}>{cluster[feat]?.toFixed(2)}</td>
                    ))}
                    <td>{cluster.count?.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default Evaluation;
