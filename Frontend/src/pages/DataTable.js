import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './DataTable.css';

const API_BASE = 'http://localhost:8000';

function DataTable() {
  const [data, setData] = useState([]);
  const [pagination, setPagination] = useState({});
  const [filters, setFilters] = useState({
    page: 1,
    page_size: 50,
    state: '',
    season: '',
    crop: '',
    sort_by: '',
    sort_order: 'asc'
  });
  const [filterOptions, setFilterOptions] = useState({});
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFilterOptions();
    fetchStats();
  }, []);

  useEffect(() => {
    fetchData();
  }, [filters]);

  const fetchFilterOptions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/data/filters`);
      setFilterOptions(res.data);
    } catch (err) {
      console.error('Error fetching filter options:', err);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/data/statistics`);
      setStats(res.data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = { ...filters };
      // Remove empty filters
      Object.keys(params).forEach(key => {
        if (params[key] === '') delete params[key];
      });

      const res = await axios.get(`${API_BASE}/data/preview`, { params });
      setData(res.data.data);
      setPagination(res.data.pagination);
    } catch (err) {
      setError('Failed to fetch data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({ ...prev, [field]: value, page: 1 }));
  };

  const handlePageChange = (newPage) => {
    setFilters(prev => ({ ...prev, page: newPage }));
  };

  const resetFilters = () => {
    setFilters({
      page: 1,
      page_size: 50,
      state: '',
      season: '',
      crop: '',
      sort_by: '',
      sort_order: 'asc'
    });
  };

  return (
    <div className="data-table-page">
      <div className="page-header">
        <h1>📊 Data Explorer</h1>
        <p>Browse and filter the crop production dataset with pagination</p>
      </div>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{stats.total_records.toLocaleString()}</div>
            <div className="stat-label">Total Records</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.unique_states}</div>
            <div className="stat-label">States</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.unique_seasons}</div>
            <div className="stat-label">Seasons</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.unique_crops}</div>
            <div className="stat-label">Unique Crops</div>
          </div>
        </div>
      )}

      <div className="card">
        <h3>🔍 Filters</h3>
        <div className="filters-grid">
          <div className="filter-group">
            <label>State</label>
            <select
              value={filters.state}
              onChange={(e) => handleFilterChange('state', e.target.value)}
            >
              <option value="">All States</option>
              {filterOptions.states?.map(state => (
                <option key={state} value={state}>{state}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Season</label>
            <select
              value={filters.season}
              onChange={(e) => handleFilterChange('season', e.target.value)}
            >
              <option value="">All Seasons</option>
              {filterOptions.seasons?.map(season => (
                <option key={season} value={season}>{season}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Crop</label>
            <select
              value={filters.crop}
              onChange={(e) => handleFilterChange('crop', e.target.value)}
            >
              <option value="">All Crops</option>
              {filterOptions.crops?.map(crop => (
                <option key={crop} value={crop}>{crop}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Sort By</label>
            <select
              value={filters.sort_by}
              onChange={(e) => handleFilterChange('sort_by', e.target.value)}
            >
              <option value="">None</option>
              {filterOptions.columns?.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Order</label>
            <select
              value={filters.sort_order}
              onChange={(e) => handleFilterChange('sort_order', e.target.value)}
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Page Size</label>
            <select
              value={filters.page_size}
              onChange={(e) => handleFilterChange('page_size', parseInt(e.target.value))}
            >
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
          </div>
        </div>
        <button className="btn btn-secondary" onClick={resetFilters} style={{marginTop: '1rem'}}>
          Reset Filters
        </button>
      </div>

      <div className="card">
        <div className="table-header">
          <h3>📋 Data Records</h3>
          {pagination.total_records && (
            <span className="records-count">
              Showing {((pagination.page - 1) * pagination.page_size) + 1} - {Math.min(pagination.page * pagination.page_size, pagination.total_records)} of {pagination.total_records.toLocaleString()}
            </span>
          )}
        </div>

        {loading ? (
          <div className="loading">Loading data...</div>
        ) : error ? (
          <div className="error">{error}</div>
        ) : (
          <>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>State</th>
                    <th>Season</th>
                    <th>Crop</th>
                    <th>N</th>
                    <th>P</th>
                    <th>K</th>
                    <th>pH</th>
                    <th>Rainfall</th>
                    <th>Temperature</th>
                    <th>Yield (ton/hec)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, idx) => (
                    <tr key={idx}>
                      <td>{row.State_Name}</td>
                      <td>{row.Season}</td>
                      <td>{row.Crop}</td>
                      <td>{row.N?.toFixed(1)}</td>
                      <td>{row.P?.toFixed(1)}</td>
                      <td>{row.K?.toFixed(1)}</td>
                      <td>{row.pH?.toFixed(2)}</td>
                      <td>{row.rainfall?.toFixed(1)}</td>
                      <td>{row.temperature?.toFixed(1)}</td>
                      <td>{row.Yield_ton_per_hec?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {pagination.total_pages > 1 && (
              <div className="pagination">
                <button
                  className="btn btn-secondary"
                  disabled={!pagination.has_prev}
                  onClick={() => handlePageChange(pagination.page - 1)}
                >
                  ← Previous
                </button>
                <span className="page-info">
                  Page {pagination.page} of {pagination.total_pages}
                </span>
                <button
                  className="btn btn-secondary"
                  disabled={!pagination.has_next}
                  onClick={() => handlePageChange(pagination.page + 1)}
                >
                  Next →
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default DataTable;
