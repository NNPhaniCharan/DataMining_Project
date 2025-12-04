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
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
  ReferenceLine,
  LineChart,
  Line,
} from "recharts";
import "./Evaluation.css";

const API_BASE = "http://localhost:8000";
const CLUSTER_COLORS = [
  "#e63946", "#f4a261", "#2a9d8f", "#264653", "#e9c46a",
  "#9b5de5", "#00bbf9", "#00f5d4", "#f15bb5"
];

function Evaluation() {
  const [yieldMetrics, setYieldMetrics] = useState(null);
  const [yieldPredictions, setYieldPredictions] = useState(null);
  const [cropMetrics, setCropMetrics] = useState(null);
  const [perClassMetrics, setPerClassMetrics] = useState(null);
  const [clusterMetrics, setClusterMetrics] = useState(null);
  const [pcaData, setPcaData] = useState(null);
  const [elbowData, setElbowData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAllMetrics();
  }, []);

  const fetchAllMetrics = async () => {
    setLoading(true);
    try {
      const [yieldRes, yieldPredRes, cropRes, perClassRes, clusterRes, pcaRes, elbowRes] = await Promise.all([
        axios.get(`${API_BASE}/evaluation/yield/metrics`),
        axios.get(`${API_BASE}/evaluation/yield/predictions`),
        axios.get(`${API_BASE}/evaluation/croprec/metrics`),
        axios.get(`${API_BASE}/evaluation/croprec/per_class_metrics`),
        axios.get(`${API_BASE}/evaluation/cluster/metrics`),
        axios.get(`${API_BASE}/evaluation/cluster/pca`),
        axios.get(`${API_BASE}/evaluation/cluster/elbow`).catch(() => ({ data: null })),
      ]);

      setYieldMetrics(yieldRes.data);
      setYieldPredictions(yieldPredRes.data);
      setCropMetrics(cropRes.data);
      setPerClassMetrics(perClassRes.data);
      setClusterMetrics(clusterRes.data);
      setPcaData(pcaRes.data);
      setElbowData(elbowRes.data);
    } catch (err) {
      setError("Failed to load evaluation metrics: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading)
    return <div className="loading">Loading evaluation metrics...</div>;
  if (error) return <div className="error">{error}</div>;

  // Prepare confusion matrix data for heatmap visualization
  const getConfusionMatrixDisplay = () => {
    if (!cropMetrics?.confusion_matrix || !cropMetrics?.classes) return null;
    
    const cm = cropMetrics.confusion_matrix;
    const classes = cropMetrics.classes;
    
    // Find classes with most errors for display
    const classErrors = classes.map((cls, i) => {
      const row = cm[i];
      const total = row.reduce((a, b) => a + b, 0);
      const correct = row[i];
      const errors = total - correct;
      return { cls, index: i, errors, total, accuracy: total > 0 ? correct / total : 0 };
    });
    
    // Sort by accuracy (worst first)
    classErrors.sort((a, b) => a.accuracy - b.accuracy);
    
    // Get top 10 most problematic classes
    const problematicClasses = classErrors.slice(0, 10);
    
    return problematicClasses;
  };

  const problematicClasses = getConfusionMatrixDisplay();

  return (
    <div className="evaluation-page">
      <div className="page-header">
        <h1>Model Evaluation Dashboard</h1>
        <p>Key visualizations and metrics for all custom algorithms</p>
      </div>

      {/* ============== YIELD PREDICTION MODEL ============== */}
      <div className="card">
        <h2>Yield Prediction - Custom Gradient Boosting</h2>
        <p className="algo-desc">{yieldMetrics?.metrics?.algorithm}</p>

        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">R² Score</div>
            <div className="metric-value">{yieldMetrics?.metrics?.r2?.toFixed(4)}</div>
            <div className="metric-subtitle">Coefficient of Determination</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">RMSE</div>
            <div className="metric-value">{yieldMetrics?.metrics?.rmse?.toFixed(4)}</div>
            <div className="metric-subtitle">Root Mean Squared Error</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">MAE</div>
            <div className="metric-value">{yieldMetrics?.metrics?.mae?.toFixed(4)}</div>
            <div className="metric-subtitle">Mean Absolute Error</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Best n_estimators</div>
            <div className="metric-value">{yieldMetrics?.metrics?.best_n_estimators}</div>
            <div className="metric-subtitle">Optimal Trees (CV)</div>
          </div>
        </div>

        {/* Actual vs Predicted Scatter Plot */}
        <div className="section-header">
          <h3>Actual vs Predicted Values</h3>
          <p>Points close to diagonal line indicate accurate predictions</p>
        </div>

        {yieldPredictions?.scatter_data && (
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis 
                  dataKey="actual" 
                  name="Actual" 
                  type="number"
                  label={{ value: 'Actual Yield (ton/ha)', position: 'bottom', offset: 40 }}
                />
                <YAxis 
                  dataKey="predicted" 
                  name="Predicted" 
                  type="number"
                  label={{ value: 'Predicted Yield (ton/ha)', angle: -90, position: 'left', offset: 40 }}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value) => value.toFixed(2)}
                />
                <ReferenceLine 
                  segment={[{ x: 0, y: 0 }, { x: 50, y: 50 }]} 
                  stroke="#e63946" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
                <Scatter 
                  data={yieldPredictions.scatter_data} 
                  fill="#4f46e5" 
                  fillOpacity={0.5}
                />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="chart-note">Red dashed line = perfect predictions (y=x). Showing {yieldPredictions.n_sampled.toLocaleString()} of {yieldPredictions.n_total.toLocaleString()} test samples.</p>
          </div>
        )}

        {/* Residual Plot */}
        <div className="section-header">
          <h3>Residual Plot</h3>
          <p>Residuals should be randomly scattered around zero</p>
        </div>

        {yieldPredictions?.residual_data && (
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis 
                  dataKey="predicted" 
                  name="Predicted" 
                  type="number"
                  label={{ value: 'Predicted Yield (ton/ha)', position: 'bottom', offset: 40 }}
                />
                <YAxis 
                  dataKey="residual" 
                  name="Residual" 
                  type="number"
                  label={{ value: 'Residual (Predicted - Actual)', angle: -90, position: 'left', offset: 40 }}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value) => value.toFixed(2)}
                />
                <ReferenceLine y={0} stroke="#e63946" strokeWidth={2} />
                <Scatter 
                  data={yieldPredictions.residual_data} 
                  fill="#10b981" 
                  fillOpacity={0.5}
                />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="chart-note">
              Mean residual: {yieldPredictions.residual_mean?.toFixed(4)} | 
              Std deviation: {yieldPredictions.residual_std?.toFixed(4)}
            </p>
          </div>
        )}

        {/* Feature Importance */}
        <div className="section-header">
          <h3>Top Feature Importances</h3>
          <p className="train-info">
            Trained on {yieldMetrics?.metrics?.n_train?.toLocaleString()} samples, 
            tested on {yieldMetrics?.metrics?.n_test?.toLocaleString()} samples
          </p>
        </div>

        <div className="coefficients-section">
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={yieldMetrics?.feature_importances?.slice(0, 12)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={120} fontSize={11} />
              <Tooltip formatter={(value) => value.toFixed(4)} />
              <Bar dataKey="importance" fill="#4f46e5" name="Importance" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ============== CROP RECOMMENDATION MODEL ============== */}
      <div className="card">
        <h2>Crop Recommendation - Custom k-NN Classifier</h2>
        <p className="algo-desc">{cropMetrics?.metrics?.algorithm}</p>

        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">Top-1 Accuracy</div>
            <div className="metric-value">{(cropMetrics?.metrics?.top1_accuracy * 100)?.toFixed(2)}%</div>
            <div className="metric-subtitle">Exact match accuracy</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Top-3 Accuracy</div>
            <div className="metric-value">{(cropMetrics?.metrics?.top3_accuracy * 100)?.toFixed(2)}%</div>
            <div className="metric-subtitle">Correct in top 3 predictions</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">F1-Score (Macro)</div>
            <div className="metric-value">{cropMetrics?.metrics?.f1_macro?.toFixed(4)}</div>
            <div className="metric-subtitle">Macro-averaged F1</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Classes</div>
            <div className="metric-value">{cropMetrics?.n_classes}</div>
            <div className="metric-subtitle">Unique crop types</div>
          </div>
        </div>

        {/* Confusion Matrix Summary */}
        <div className="section-header">
          <h3>Confusion Matrix Analysis</h3>
          <p>Classes with lowest accuracy (most challenging to classify)</p>
        </div>

        {problematicClasses && (
          <div className="confusion-summary">
            <table className="confusion-table">
              <thead>
                <tr>
                  <th>Crop</th>
                  <th>Accuracy</th>
                  <th>Correct</th>
                  <th>Total</th>
                  <th>Visual</th>
                </tr>
              </thead>
              <tbody>
                {problematicClasses.map((item, idx) => (
                  <tr key={idx}>
                    <td className="crop-name">{item.cls}</td>
                    <td className={item.accuracy < 0.8 ? 'low-accuracy' : 'good-accuracy'}>
                      {(item.accuracy * 100).toFixed(1)}%
                    </td>
                    <td>{cropMetrics.confusion_matrix[item.index][item.index]}</td>
                    <td>{item.total}</td>
                    <td>
                      <div className="accuracy-bar">
                        <div 
                          className="accuracy-fill" 
                          style={{ 
                            width: `${item.accuracy * 100}%`,
                            backgroundColor: item.accuracy < 0.5 ? '#e63946' : item.accuracy < 0.8 ? '#f4a261' : '#2a9d8f'
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Per-Class Metrics Bar Chart */}
        <div className="section-header">
          <h3>Per-Class Performance (Worst 15)</h3>
          <p>Precision, Recall, and F1-Score for most challenging crops</p>
        </div>

        {perClassMetrics?.per_class_metrics && (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart 
              data={perClassMetrics.per_class_metrics.slice(0, 15)} 
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis dataKey="class" type="category" width={90} fontSize={11} />
              <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="precision" fill="#4f46e5" name="Precision" />
              <Bar dataKey="recall" fill="#10b981" name="Recall" />
              <Bar dataKey="f1" fill="#f59e0b" name="F1-Score" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ============== CLUSTERING MODEL ============== */}
      <div className="card">
        <h2>Cluster Assignment - Custom K-Means</h2>
        <p className="algo-desc">{clusterMetrics?.metrics?.algorithm}</p>

        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">Number of Clusters</div>
            <div className="metric-value">{clusterMetrics?.metrics?.n_clusters}</div>
            <div className="metric-subtitle">k-means++ initialization</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Silhouette Score</div>
            <div className="metric-value">{clusterMetrics?.metrics?.silhouette_score?.toFixed(4)}</div>
            <div className="metric-subtitle">Cluster cohesion [-1, 1]</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Calinski-Harabasz</div>
            <div className="metric-value">{clusterMetrics?.metrics?.calinski_harabasz_score?.toFixed(0)}</div>
            <div className="metric-subtitle">Variance ratio (higher = better)</div>
          </div>
        </div>

        {/* PCA Scatter Plot */}
        <div className="section-header">
          <h3>PCA Cluster Visualization</h3>
          <p>2D projection showing cluster separation ({pcaData?.total_variance_explained ? (pcaData.total_variance_explained * 100).toFixed(1) : '?'}% variance explained)</p>
        </div>

        {pcaData?.scatter_data && (
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={450}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis 
                  dataKey="x" 
                  name="PC1" 
                  type="number"
                  label={{ value: `PC1 (${pcaData.explained_variance?.[0] ? (pcaData.explained_variance[0] * 100).toFixed(1) : '?'}% variance)`, position: 'bottom', offset: 40 }}
                />
                <YAxis 
                  dataKey="y" 
                  name="PC2" 
                  type="number"
                  label={{ value: `PC2 (${pcaData.explained_variance?.[1] ? (pcaData.explained_variance[1] * 100).toFixed(1) : '?'}% variance)`, angle: -90, position: 'left', offset: 40 }}
                />
                <ZAxis dataKey="cluster" range={[30, 30]} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ payload }) => {
                    if (payload && payload.length > 0) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-tooltip">
                          <p><strong>Cluster {data.cluster}</strong></p>
                          <p>PC1: {data.x.toFixed(3)}</p>
                          <p>PC2: {data.y.toFixed(3)}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter data={pcaData.scatter_data}>
                  {pcaData.scatter_data.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                      fillOpacity={0.6}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <p className="chart-note">
              Showing {pcaData.n_sampled?.toLocaleString()} of {pcaData.n_total?.toLocaleString()} data points (stratified sample)
            </p>
            
            {/* Cluster Legend */}
            <div className="cluster-legend">
              {clusterMetrics?.metrics?.counts?.map((count, idx) => (
                <div key={idx} className="legend-item">
                  <span 
                    className="legend-color" 
                    style={{ backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] }}
                  />
                  <span>Cluster {idx} ({count.toLocaleString()})</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Elbow Method Analysis */}
        {elbowData && (
          <>
            <div className="section-header">
              <h3>Elbow Method Analysis</h3>
              <p>Determining optimal number of clusters (k) using inertia and silhouette scores</p>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem", marginBottom: "2rem" }}>
              {/* Inertia (WCSS) Chart */}
              <div>
                <h4 style={{ fontSize: "0.95rem", marginBottom: "0.75rem", color: "#333" }}>
                  Inertia (Within-Cluster Sum of Squares)
                </h4>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={elbowData.elbow_data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis
                      dataKey="k"
                      label={{ value: "Number of Clusters (k)", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis
                      label={{ value: "Inertia", angle: -90, position: "insideLeft" }}
                      tickFormatter={(v) => v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v}
                    />
                    <Tooltip
                      formatter={(value) => [value.toLocaleString(), "Inertia"]}
                      labelFormatter={(label) => `k = ${label}`}
                    />
                    <ReferenceLine
                      x={elbowData.selected_k}
                      stroke="#e63946"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={{ value: `k=${elbowData.selected_k}`, position: "top", fill: "#e63946", fontSize: 12 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="inertia"
                      stroke="#4f46e5"
                      strokeWidth={3}
                      dot={{ fill: "#4f46e5", r: 5, strokeWidth: 2, stroke: "#fff" }}
                      activeDot={{ r: 8, fill: "#3730a3" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Silhouette Score Chart */}
              <div>
                <h4 style={{ fontSize: "0.95rem", marginBottom: "0.75rem", color: "#333" }}>
                  Silhouette Score (Higher = Better Separation)
                </h4>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={elbowData.elbow_data.filter((d) => d.k > 2)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis
                      dataKey="k"
                      label={{ value: "Number of Clusters (k)", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis
                      domain={[0, "auto"]}
                      label={{ value: "Silhouette", angle: -90, position: "insideLeft" }}
                      tickFormatter={(v) => v.toFixed(2)}
                    />
                    <Tooltip
                      formatter={(value) => [value.toFixed(4), "Silhouette"]}
                      labelFormatter={(label) => `k = ${label}`}
                    />
                    <ReferenceLine
                      x={elbowData.selected_k}
                      stroke="#e63946"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={{ value: `k=${elbowData.selected_k}`, position: "top", fill: "#e63946", fontSize: 12 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="silhouette"
                      stroke="#10b981"
                      strokeWidth={3}
                      dot={{ fill: "#10b981", r: 5, strokeWidth: 2, stroke: "#fff" }}
                      activeDot={{ r: 8, fill: "#059669" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Summary Stats */}
            <div className="metrics-grid" style={{ marginBottom: "2rem" }}>
              <div className="metric-box">
                <div className="metric-label">Selected k</div>
                <div className="metric-value" style={{ color: "#e63946" }}>{elbowData.selected_k}</div>
                <div className="metric-subtitle">Optimal clusters</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Inertia at k={elbowData.selected_k}</div>
                <div className="metric-value" style={{ color: "#4f46e5" }}>
                  {elbowData.elbow_data.find((d) => d.k === elbowData.selected_k)?.inertia.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </div>
                <div className="metric-subtitle">Within-cluster variance</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Silhouette at k={elbowData.selected_k}</div>
                <div className="metric-value" style={{ color: "#10b981" }}>
                  {elbowData.elbow_data.find((d) => d.k === elbowData.selected_k)?.silhouette.toFixed(4)}
                </div>
                <div className="metric-subtitle">Cluster separation quality</div>
              </div>
            </div>
          </>
        )}

        {/* Cluster Sizes Bar Chart */}
        <div className="section-header">
          <h3>Cluster Size Distribution</h3>
        </div>

        {clusterMetrics?.metrics?.counts && (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart 
              data={clusterMetrics.metrics.counts.map((count, idx) => ({
                name: `Cluster ${idx}`,
                count: count,
                percentage: (count / clusterMetrics.metrics.counts.reduce((a, b) => a + b, 0) * 100).toFixed(1)
              }))}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [value.toLocaleString(), 'Count']}
                labelFormatter={(label) => label}
              />
              <Bar dataKey="count" name="Sample Count">
                {clusterMetrics.metrics.counts.map((_, idx) => (
                  <Cell key={`cell-${idx}`} fill={CLUSTER_COLORS[idx % CLUSTER_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}

        {/* Quality Info */}
        <div className="quality-info">
          <h4>Clustering Quality Interpretation</h4>
          <p>
            <strong>Silhouette Score ({clusterMetrics?.metrics?.silhouette_score?.toFixed(3)}):</strong> Values range from -1 to 1. 
            Score &gt; 0.5 indicates strong structure, 0.25-0.5 indicates reasonable structure. 
            Your score suggests {clusterMetrics?.metrics?.silhouette_score > 0.5 ? 'well-separated' : clusterMetrics?.metrics?.silhouette_score > 0.25 ? 'reasonably separated' : 'overlapping'} clusters.
          </p>
          <p>
            <strong>Calinski-Harabasz ({clusterMetrics?.metrics?.calinski_harabasz_score?.toFixed(0)}):</strong> Higher values indicate 
            better-defined clusters. Your score indicates {clusterMetrics?.metrics?.calinski_harabasz_score > 10000 ? 'well-defined' : 'moderately defined'} cluster boundaries.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Evaluation;
