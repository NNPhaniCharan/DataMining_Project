import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import './App.css';
import DataTable from './pages/DataTable';
import Evaluation from './pages/Evaluation';
import YieldPrediction from './pages/YieldPrediction';
import CropRecommendation from './pages/CropRecommendation';
import ClusterAssignment from './pages/ClusterAssignment';

function Navigation() {
  const location = useLocation();
  
  const isActive = (path) => location.pathname === path;
  
  return (
    <nav className="navbar">
      <div className="nav-brand">
        <h2>Agri Analytics</h2>
        <p>Custom ML Algorithms</p>
      </div>
      <div className="nav-links">
        <Link to="/" className={isActive('/') ? 'active' : ''}>
          Data Explorer
        </Link>
        <Link to="/evaluation" className={isActive('/evaluation') ? 'active' : ''}>
          Model Evaluation
        </Link>
        <Link to="/yield" className={isActive('/yield') ? 'active' : ''}>
          Yield Prediction
        </Link>
        <Link to="/crop" className={isActive('/crop') ? 'active' : ''}>
          Crop Recommendation
        </Link>
        <Link to="/cluster" className={isActive('/cluster') ? 'active' : ''}>
          Cluster Assignment
        </Link>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<DataTable />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/yield" element={<YieldPrediction />} />
            <Route path="/crop" element={<CropRecommendation />} />
            <Route path="/cluster" element={<ClusterAssignment />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
