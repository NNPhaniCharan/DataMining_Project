import React, { useState, useEffect } from "react";
import axios from "axios";
import "./Prediction.css";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

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
  const [contextStats, setContextStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overview"); // overview, recommendations

  useEffect(() => {
    fetchFilterOptions();
  }, []);

  useEffect(() => {
    // Fetch context stats when crop/state/season changes
    if (formData.Crop && formData.State_Name && formData.Season) {
      fetchContextStats();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [formData.Crop, formData.State_Name, formData.Season]);

  const fetchFilterOptions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/data/filters`);
      setFilterOptions(res.data);
    } catch (err) {
      console.error("Error fetching filter options:", err);
    }
  };

  const fetchContextStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/data/context-stats`, {
        params: {
          state: formData.State_Name,
          season: formData.Season,
          crop: formData.Crop,
        },
      });
      setContextStats(res.data);
    } catch (err) {
      console.error("Error fetching context stats:", err);
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

  // Helper function to prepare yield distribution data
  const prepareYieldDistributionData = () => {
    if (!contextStats || !result) return [];

    const dist = contextStats.yield_distribution;
    return [
      { category: "Min", value: dist.min, fill: "#e0e0e0" },
      { category: "Q1", value: dist.q25, fill: "#ffb74d" },
      { category: "Median", value: dist.median, fill: "#4caf50" },
      { category: "Q3", value: dist.q75, fill: "#ffb74d" },
      { category: "Max", value: dist.max, fill: "#e0e0e0" },
      {
        category: "Your Prediction",
        value: result.predicted_yield_ton_per_hec,
        fill: "#2196f3",
      },
    ];
  };

  // Helper function to get yield category and color
  const getYieldCategory = (yieldValue) => {
    if (yieldValue < 2)
      return { category: "Low", color: "#ff6b6b", percent: 33 };
    if (yieldValue < 4)
      return { category: "Medium", color: "#ffd93d", percent: 66 };
    return { category: "High", color: "#6bcf7f", percent: 100 };
  };

  // Generate actionable recommendations
  const generateRecommendations = () => {
    if (!contextStats || !result) return [];

    const recommendations = [];
    const features = ["N", "P", "K", "pH", "rainfall", "temperature"];
    const currentYield = result.predicted_yield_ton_per_hec;
    const medianYield = contextStats.yield_distribution.median;

    // Compare inputs to optimal ranges
    features.forEach((feature) => {
      const userValue = formData[feature];
      const cropAvg = contextStats.context_averages[feature];
      const diff = ((userValue - cropAvg) / cropAvg) * 100;

      if (Math.abs(diff) > 15) {
        let recommendation = {
          feature: feature,
          status: diff > 0 ? "high" : "low",
          currentValue: userValue,
          recommendedValue: cropAvg.toFixed(1),
          impact: Math.abs(diff) > 30 ? "high" : "medium",
          message: "",
        };

        if (diff < -15) {
          recommendation.message = `Your ${feature} level is ${Math.abs(
            diff
          ).toFixed(
            0
          )}% below optimal. Consider increasing to ${cropAvg.toFixed(
            1
          )} for better yield.`;
        } else if (diff > 15) {
          recommendation.message = `Your ${feature} level is ${diff.toFixed(
            0
          )}% above optimal. This may not improve yield further and could be wasteful.`;
        }

        recommendations.push(recommendation);
      }
    });

    // Yield-based recommendation
    if (currentYield < medianYield) {
      const gap = ((medianYield - currentYield) / medianYield) * 100;
      recommendations.unshift({
        feature: "Overall",
        status: "info",
        impact: gap > 30 ? "high" : "medium",
        message: `Your predicted yield is ${gap.toFixed(
          0
        )}% below the median for ${
          formData.Crop
        }. Review nutrient levels and environmental factors.`,
      });
    } else if (currentYield > contextStats.yield_distribution.q75) {
      recommendations.unshift({
        feature: "Overall",
        status: "success",
        impact: "positive",
        message: `Excellent! Your predicted yield is in the top 25% for ${formData.Crop}. Your input parameters are well-optimized.`,
      });
    }

    return recommendations;
  };

  // Prepare data for radar chart
  const prepareRadarData = () => {
    if (!contextStats) return [];

    const features = ["N", "P", "K", "pH", "rainfall", "temperature"];
    return features.map((feature) => {
      const cropAvg = contextStats.context_averages[feature];
      const globalAvg = contextStats.global_averages[feature];
      const userValue = formData[feature];

      // Normalize to percentage of crop average
      return {
        feature: feature,
        "Your Input": cropAvg > 0 ? (userValue / cropAvg) * 100 : 100,
        Optimal: 100,
        "Global Avg": cropAvg > 0 ? (globalAvg / cropAvg) * 100 : 100,
      };
    });
  };

  // Prepare detailed comparison table data
  const prepareDetailedComparison = () => {
    if (!contextStats) return [];

    const features = ["N", "P", "K", "pH", "rainfall", "temperature"];
    return features.map((feature) => {
      const userValue = formData[feature];
      const cropAvg = contextStats.context_averages[feature];
      const globalAvg = contextStats.global_averages[feature];
      const diffPct = ((userValue - cropAvg) / cropAvg) * 100;

      return {
        feature: feature,
        yourValue: userValue.toFixed(1),
        cropAverage: cropAvg.toFixed(1),
        globalAverage: globalAvg.toFixed(1),
        difference: diffPct.toFixed(1),
        status:
          Math.abs(diffPct) < 10
            ? "optimal"
            : Math.abs(diffPct) < 25
            ? "acceptable"
            : "suboptimal",
      };
    });
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

                {contextStats && (
                  <div style={{ fontSize: "0.875rem", color: "#666" }}>
                    {result.predicted_yield_ton_per_hec >
                    contextStats.yield_distribution.median
                      ? `✓ ${(
                          (result.predicted_yield_ton_per_hec /
                            contextStats.yield_distribution.median -
                            1) *
                          100
                        ).toFixed(0)}% above median`
                      : `↓ ${(
                          (1 -
                            result.predicted_yield_ton_per_hec /
                              contextStats.yield_distribution.median) *
                          100
                        ).toFixed(0)}% below median`}
                    {" for "}
                    {formData.Crop} in {formData.State_Name}
                  </div>
                )}
              </div>

              {/* Tabs for different analysis views */}
              <div style={{ marginBottom: "1.5rem" }}>
                <div
                  style={{
                    display: "flex",
                    gap: "0.5rem",
                    borderBottom: "2px solid #f0f0f0",
                  }}
                >
                  <button
                    onClick={() => setActiveTab("overview")}
                    style={{
                      padding: "0.75rem 1.5rem",
                      border: "none",
                      background: "none",
                      cursor: "pointer",
                      fontWeight: activeTab === "overview" ? "600" : "400",
                      color: activeTab === "overview" ? "#2196f3" : "#666",
                      borderBottom:
                        activeTab === "overview"
                          ? "3px solid #2196f3"
                          : "3px solid transparent",
                      transition: "all 0.2s",
                    }}
                  >
                    📊 Overview
                  </button>
                  <button
                    onClick={() => setActiveTab("recommendations")}
                    style={{
                      padding: "0.75rem 1.5rem",
                      border: "none",
                      background: "none",
                      cursor: "pointer",
                      fontWeight:
                        activeTab === "recommendations" ? "600" : "400",
                      color:
                        activeTab === "recommendations" ? "#2196f3" : "#666",
                      borderBottom:
                        activeTab === "recommendations"
                          ? "3px solid #2196f3"
                          : "3px solid transparent",
                      transition: "all 0.2s",
                    }}
                  >
                    💡 Recommendations
                  </button>
                </div>
              </div>

              {/* OVERVIEW TAB */}
              {activeTab === "overview" && contextStats && (
                <div>
                  {/* Radar Chart - Input Profile */}
                  <div style={{ marginBottom: "2rem" }}>
                    <h4
                      style={{
                        fontSize: "0.875rem",
                        color: "#666",
                        marginBottom: "0.5rem",
                      }}
                    >
                      🎯 Input Profile Analysis
                    </h4>
                    <p
                      style={{
                        fontSize: "0.75rem",
                        color: "#999",
                        marginBottom: "1rem",
                      }}
                    >
                      Your inputs relative to optimal {formData.Crop} conditions
                      (100% = optimal)
                    </p>
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={prepareRadarData()}>
                        <PolarGrid stroke="#e0e0e0" />
                        <PolarAngleAxis
                          dataKey="feature"
                          tick={{ fill: "#666", fontSize: 11 }}
                        />
                        <PolarRadiusAxis
                          angle={90}
                          domain={[0, 150]}
                          tick={{ fill: "#666", fontSize: 10 }}
                        />
                        <Radar
                          name="Your Input"
                          dataKey="Your Input"
                          stroke="#2196f3"
                          fill="#2196f3"
                          fillOpacity={0.5}
                        />
                        <Radar
                          name="Optimal"
                          dataKey="Optimal"
                          stroke="#4caf50"
                          fill="#4caf50"
                          fillOpacity={0.3}
                        />
                        <Legend />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Detailed Comparison Table */}
                  <div style={{ marginBottom: "2rem" }}>
                    <h4
                      style={{
                        fontSize: "0.875rem",
                        color: "#666",
                        marginBottom: "0.5rem",
                      }}
                    >
                      📊 Detailed Input Comparison
                    </h4>
                    <p
                      style={{
                        fontSize: "0.75rem",
                        color: "#999",
                        marginBottom: "1rem",
                      }}
                    >
                      How each parameter compares (
                      {contextStats.context_count.toLocaleString()}{" "}
                      {formData.Crop} samples analyzed)
                    </p>
                    <div
                      style={{
                        overflowX: "auto",
                        fontSize: "0.8rem",
                      }}
                    >
                      <table
                        style={{
                          width: "100%",
                          borderCollapse: "collapse",
                          textAlign: "left",
                        }}
                      >
                        <thead>
                          <tr style={{ borderBottom: "2px solid #e0e0e0" }}>
                            <th
                              style={{ padding: "0.75rem", fontWeight: "600" }}
                            >
                              Parameter
                            </th>
                            <th
                              style={{ padding: "0.75rem", fontWeight: "600" }}
                            >
                              Your Value
                            </th>
                            <th
                              style={{ padding: "0.75rem", fontWeight: "600" }}
                            >
                              Crop Avg
                            </th>
                            <th
                              style={{ padding: "0.75rem", fontWeight: "600" }}
                            >
                              Difference
                            </th>
                            <th
                              style={{ padding: "0.75rem", fontWeight: "600" }}
                            >
                              Status
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {prepareDetailedComparison().map((row, idx) => (
                            <tr
                              key={idx}
                              style={{ borderBottom: "1px solid #f0f0f0" }}
                            >
                              <td
                                style={{
                                  padding: "0.75rem",
                                  fontWeight: "500",
                                }}
                              >
                                {row.feature}
                              </td>
                              <td style={{ padding: "0.75rem" }}>
                                {row.yourValue}
                              </td>
                              <td style={{ padding: "0.75rem" }}>
                                {row.cropAverage}
                              </td>
                              <td
                                style={{
                                  padding: "0.75rem",
                                  color:
                                    Math.abs(parseFloat(row.difference)) < 10
                                      ? "#4caf50"
                                      : Math.abs(parseFloat(row.difference)) <
                                        25
                                      ? "#ff9800"
                                      : "#f44336",
                                  fontWeight: "600",
                                }}
                              >
                                {row.difference > 0 ? "+" : ""}
                                {row.difference}%
                              </td>
                              <td style={{ padding: "0.75rem" }}>
                                <span
                                  style={{
                                    padding: "0.25rem 0.5rem",
                                    borderRadius: "0.25rem",
                                    fontSize: "0.75rem",
                                    fontWeight: "600",
                                    background:
                                      row.status === "optimal"
                                        ? "#4caf5020"
                                        : row.status === "acceptable"
                                        ? "#ff980020"
                                        : "#f4433620",
                                    color:
                                      row.status === "optimal"
                                        ? "#4caf50"
                                        : row.status === "acceptable"
                                        ? "#ff9800"
                                        : "#f44336",
                                  }}
                                >
                                  {row.status === "optimal"
                                    ? "✓ Optimal"
                                    : row.status === "acceptable"
                                    ? "~ OK"
                                    : "⚠ Review"}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Yield Benchmarking */}
                  <div style={{ marginBottom: "2rem" }}>
                    <h4
                      style={{
                        fontSize: "0.875rem",
                        color: "#666",
                        marginBottom: "0.5rem",
                      }}
                    >
                      📈 Yield Benchmarking
                    </h4>
                    <p
                      style={{
                        fontSize: "0.75rem",
                        color: "#999",
                        marginBottom: "1rem",
                      }}
                    >
                      Your prediction vs {formData.Crop} yield distribution
                    </p>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={prepareYieldDistributionData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis
                          dataKey="category"
                          tick={{ fill: "#666", fontSize: 11 }}
                        />
                        <YAxis
                          tick={{ fill: "#666", fontSize: 11 }}
                          label={{
                            value: "Yield (t/ha)",
                            angle: -90,
                            position: "insideLeft",
                          }}
                        />
                        <Tooltip
                          formatter={(value) => `${value.toFixed(2)} t/ha`}
                        />
                        <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                          {prepareYieldDistributionData().map(
                            (entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            )
                          )}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* RECOMMENDATIONS TAB */}
              {activeTab === "recommendations" && contextStats && (
                <div>
                  {generateRecommendations().length > 0 ? (
                    <>
                      <div style={{ marginBottom: "1.5rem" }}>
                        <h4
                          style={{
                            fontSize: "0.875rem",
                            color: "#666",
                            marginBottom: "0.5rem",
                          }}
                        >
                          💡 Actionable Insights
                        </h4>
                        <p
                          style={{
                            fontSize: "0.75rem",
                            color: "#999",
                            marginBottom: "1rem",
                          }}
                        >
                          Recommendations to optimize your yield based on
                          analysis
                        </p>
                      </div>

                      {generateRecommendations().map((rec, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "1rem",
                            marginBottom: "1rem",
                            background:
                              rec.status === "success"
                                ? "#4caf5010"
                                : rec.status === "high" || rec.status === "low"
                                ? "#ff980010"
                                : "#2196f310",
                            border: `1px solid ${
                              rec.status === "success"
                                ? "#4caf50"
                                : rec.status === "high" || rec.status === "low"
                                ? "#ff9800"
                                : "#2196f3"
                            }`,
                            borderRadius: "0.5rem",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              alignItems: "flex-start",
                              gap: "0.75rem",
                            }}
                          >
                            <div
                              style={{
                                fontSize: "1.25rem",
                                flexShrink: 0,
                              }}
                            >
                              {rec.status === "success"
                                ? "✓"
                                : rec.status === "info"
                                ? "ℹ"
                                : "⚠"}
                            </div>
                            <div style={{ flex: 1 }}>
                              <div
                                style={{
                                  fontWeight: "600",
                                  fontSize: "0.875rem",
                                  marginBottom: "0.25rem",
                                  color: "#333",
                                }}
                              >
                                {rec.feature}
                                {rec.impact && rec.impact !== "positive" && (
                                  <span
                                    style={{
                                      marginLeft: "0.5rem",
                                      padding: "0.125rem 0.5rem",
                                      background:
                                        rec.impact === "high"
                                          ? "#f4433620"
                                          : "#ff980020",
                                      color:
                                        rec.impact === "high"
                                          ? "#f44336"
                                          : "#ff9800",
                                      borderRadius: "0.25rem",
                                      fontSize: "0.75rem",
                                    }}
                                  >
                                    {rec.impact} priority
                                  </span>
                                )}
                              </div>
                              <div
                                style={{
                                  fontSize: "0.85rem",
                                  color: "#555",
                                  lineHeight: "1.5",
                                }}
                              >
                                {rec.message}
                              </div>
                              {rec.currentValue && rec.recommendedValue && (
                                <div
                                  style={{
                                    marginTop: "0.5rem",
                                    padding: "0.5rem",
                                    background: "#fff",
                                    borderRadius: "0.25rem",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  <span style={{ color: "#999" }}>
                                    Current:
                                  </span>{" "}
                                  <strong>{rec.currentValue}</strong>
                                  <span
                                    style={{
                                      margin: "0 0.5rem",
                                      color: "#999",
                                    }}
                                  >
                                    →
                                  </span>
                                  <span style={{ color: "#999" }}>
                                    Optimal:
                                  </span>{" "}
                                  <strong style={{ color: "#4caf50" }}>
                                    {rec.recommendedValue}
                                  </strong>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}

                      {/* Additional Context */}
                      <div
                        style={{
                          marginTop: "2rem",
                          padding: "1rem",
                          background: "#f8f9fa",
                          borderRadius: "0.5rem",
                          fontSize: "0.8rem",
                          color: "#666",
                        }}
                      >
                        <p style={{ marginBottom: "0.5rem" }}>
                          <strong>Context:</strong>
                        </p>
                        <ul
                          style={{ marginLeft: "1.25rem", lineHeight: "1.6" }}
                        >
                          <li>
                            Analysis based on{" "}
                            {contextStats.context_count.toLocaleString()}{" "}
                            similar {formData.Crop} samples
                          </li>
                          <li>
                            Current yield prediction:{" "}
                            {result.predicted_yield_ton_per_hec.toFixed(3)} t/ha
                          </li>
                          <li>
                            Median yield for {formData.Crop}:{" "}
                            {contextStats.yield_distribution.median.toFixed(3)}{" "}
                            t/ha
                          </li>
                          <li>
                            Top quartile yield:{" "}
                            {contextStats.yield_distribution.q75.toFixed(3)}{" "}
                            t/ha
                          </li>
                        </ul>
                      </div>
                    </>
                  ) : (
                    <div
                      style={{
                        textAlign: "center",
                        padding: "3rem 1rem",
                        background: "#f8f9fa",
                        borderRadius: "0.5rem",
                      }}
                    >
                      <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>
                        🎉
                      </div>
                      <div
                        style={{
                          fontSize: "1rem",
                          fontWeight: "600",
                          marginBottom: "0.5rem",
                        }}
                      >
                        Excellent Configuration!
                      </div>
                      <div style={{ fontSize: "0.875rem", color: "#666" }}>
                        Your input parameters are well-aligned with optimal
                        conditions for {formData.Crop}. No major adjustments
                        needed.
                      </div>
                    </div>
                  )}
                </div>
              )}

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
