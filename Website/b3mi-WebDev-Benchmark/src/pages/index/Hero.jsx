import './Hero.css';
import React, { useState, useEffect } from 'react';

const Hero = () => {

  const [data, setData] = useState([]);
  const resolution = '224x224';
  const dataset = 'MedMNIST+';

  useEffect(() => {
    // Fetch top 5 leaderboard entries
    fetchLeaderboard();
  }, []);

  const fetchLeaderboard = () => {
    const params = new URLSearchParams();
    params.append('resolution', resolution);
    params.append('metric', 'auc_test');
    params.append("dataset", dataset);

    fetch(`http://localhost:5000/leaderboard?${params.toString()}`, {
      method: 'GET',
      credentials: 'include',
    })
      .then(response => response.json())
      .then(result => {
        if (result.status === 'success') {
          // Take only the top 5 entries
          const topFive = result.data.slice(0, 5);
          setData(topFive);
        } else {
          setData([]);
        }
      })
      .catch(error => {
        console.error('Error fetching leaderboard:', error);
      });
  };

  return (
    <div className="hero-container">
      <div className="HeroTableContainer">
        <h2>Leaderboard</h2>
        <table className="table" id="HeroTable">
          <thead>
            <tr>
              <th scope="col">Rank</th>
              <th scope="col">Team Name</th>
              <th scope="col">Model Name</th>
              <th scope="col">AUC</th>
            </tr>
          </thead>
          <tbody>
            {data.length > 0 ? (
              data.map((entry, index) => (
                <tr key={entry.id}>
                  <th scope="row">{index + 1}</th>
                  <td>{entry.teamname}</td>
                  <td>{entry.model_name}</td>
                  <td>{entry.metric_value !== null ? entry.metric_value.toFixed(4) : '-'}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="4">No data available</td>
              </tr>
            )}
          </tbody>
        </table>
        <p className="leaderboard-description">
          The above leaderboard shows the top 5 entries for the AUC metric of models trained on images with resolution {resolution}.
        </p>
      </div>
      <div className="hero-text">
        <h1 id="HeroHeader">BIG Benchmark</h1>
        <p id="HeroText">
          The Biomedical Image Generalization (BIG) Benchmark out of the Otto-Friedrich University of Bamberg
          aims to provide a challenge for AI researchers to compare their models
          on a diverse set of medical images against a baseline as well as against other
          researchers. The underlying dataset consists of the MedMNIST+
          collection, thus offering images from several biomedical domains.The BIG Benchmark aims to investigate
          the generalization capability of models. This challenge is open to anyone interested in
          furthering the field of Artificial Intelligence and building a
          foundation for medical providers to take advantage of the modern
          solutions to better patient care and take strain off of healthcare
          professionals.
        </p>
        <a href={"./dataset"} className="hero-button">
          {"Dataset"}
        </a>
        <a href={"./leaderboard"} className="hero-button">
          {"Leaderboard"}
        </a>
      </div>
    </div>
  );
};

export default Hero;
