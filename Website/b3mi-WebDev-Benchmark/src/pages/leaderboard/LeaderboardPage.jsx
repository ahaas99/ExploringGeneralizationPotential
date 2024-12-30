import React, { useState, useEffect } from 'react';
import NavBar from "../../components/NavBar";
import './Leaderboard.css'; // Import the CSS file

function Leaderboard() {
  // Set default resolution to '224x224'
  const [resolution, setResolution] = useState('224x224');
  const [metric, setMetric] = useState('auc_test');  // Default metric
    const [data, setData] = useState([]);
    const [dataset, setDataset] = useState('MedMNIST+');

  // Remove 'All' option from the resolutions array
    const resolutions = ['28x28', '64x64', '128x128', '224x224'];
    const datasets = ['MedMNIST+', 'MedMNIST-C'];

  const metrics = [
    { value: 'auc_test', label: 'AUC' },
    { value: 'balacc_test', label: 'Balanced Accuracy' },
    { value: 'co_test', label: "Cohen's Kappa" },
  ];

  useEffect(() => {
    fetchLeaderboard();
  }, [resolution, metric, dataset]);

  const fetchLeaderboard = () => {
    const params = new URLSearchParams();
    params.append('resolution', resolution);  // Resolution will always have a value
    params.append('metric', metric);
    params.append('dataset', dataset);

    fetch(`http://localhost:5000/leaderboard?${params.toString()}`, {
      method: 'GET',
      credentials: 'include',
    })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          setData(data.data);
        } else {
          setData([]);
        }
      })
      .catch(error => {
        console.error('Error fetching leaderboard:', error);
      });
  };

  return (
    <>
      <NavBar />
      <div className="container mt-4">
        <h1 className="leaderboard-title">Leaderboard</h1>
        <div className="row mb-4">
          <div className="col-md-6">
            <label htmlFor="resolutionSelect" className="form-label">Resolution:</label>
            <select
              id="resolutionSelect"
              className="form-select"
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
            >
              {resolutions.map((res) => (
                <option key={res} value={res}>{res}</option>
              ))}
            </select>
          </div>
          <div className="col-md-6">
            <label htmlFor="metricSelect" className="form-label">Metric:</label>
            <select
              id="metricSelect"
              className="form-select"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
            >
              {metrics.map((metricOption) => (
                <option key={metricOption.value} value={metricOption.value}>{metricOption.label}</option>
              ))}
            </select>
                  </div>
                  <div className="col-md-6 mx-auto">
                      <label htmlFor="resolutionSelect" className="form-label">Dataset:</label>
                      <select
                          id="resolutionSelect"
                          className="form-select"
                          value={dataset}
                          onChange={(e) => setDataset(e.target.value)}
                      >
                          {datasets.map((ds) => (
                              <option key={ds} value={ds}>{ds}</option>
                          ))}
                      </select>
                  </div>
        </div>

        <div className="leaderboard-container">
          {data.length > 0 ? (
            <table className="table leaderboard-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Team Name</th>
                  <th>Study Link</th>
                  <th>Model Name</th>
                  <th>Description</th>
                  <th>Metric Value</th>
                </tr>
              </thead>
              <tbody>
                {data.map((submission) => (
                  <tr key={submission.id}>
                    <td>{submission.id}</td>
                    <td>{submission.teamname}</td>
                    <td>
                      {submission.study_link !== '-' ? (
                        <a href={submission.study_link} target="_blank" rel="noopener noreferrer">Link</a>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td>{submission.model_name}</td>
                    <td>{submission.description}</td>
                    <td>{submission.metric_value !== null ? submission.metric_value.toFixed(4) : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div>
              <h2 className="text-center">No matching submissions</h2>
              <table className="table leaderboard-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Team Name</th>
                    <th>Study Link</th>
                    <th>Model Name</th>
                    <th>Description</th>
                    <th>Metric Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default Leaderboard;
