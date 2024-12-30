import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import './UserProfile.css';
import NavBar from '../../components/NavBar';
import Spinner from 'react-bootstrap/Spinner';
import { saveAs } from 'file-saver'; // Import file-saver for downloading files

function Profile() {
    const [profileData, setProfileData] = useState(null);  // Initialize profileData as null
    const [studyLink, setStudyLink] = useState('');
    const [file, setFile] = useState();
    const [modelName, setModelName] = useState('');
    const [description, setDescription] = useState('');
    const [resolution, setResolution] = useState('28x28');
    const [submissions, setSubmissions] = useState([]); // For the list of submissions made by the user
    const [submissionId1, setSubmissionId1] = useState('');
    const [submissionId2, setSubmissionId2] = useState('');
    const [dataset, setDataset] = useState('MedMNIST+');

    const navigate = useNavigate();

    useEffect(() => {
      // Fetch user profile data from the backend
      fetch('http://localhost:5000/profile', {
          method: 'GET',
          credentials: 'include',
      })
      .then(response => response.json())
      .then(data => {
          if (data.status === "failed") {
              // Redirect to login if not logged in
              navigate('/login');
          } else {
              setProfileData(data);

              // Fetch the user submissions
              fetch('http://localhost:5000/user_submissions', {
                  method: 'GET',
                  credentials: 'include',
              })
              .then(response => response.json())
              .then(submissionData => {
                  if (submissionData.status === 'success') {
                      setSubmissions(submissionData.submissions);
                  } else {
                      console.error('Failed to fetch submissions:', submissionData.message);
                  }
              })
              .catch((error) => {
                  console.error('Error fetching submissions:', error);
              });
          }
      })
      .catch((error) => {
          console.error('Error fetching profile:', error);
      });
  }, [navigate]);

    if (!profileData) {
        return (
            <div>
                <Spinner animation="border" />
                <div>Loading Profile Data...</div>;
            </div>
        );
    }

    const handleOnChange = (e) => {
        setFile(e.target.files[0]);
      };

    const handleCSVUpload = (e) => {
        e.preventDefault();
    
        const formData = new FormData();
        formData.append('file', file);
        formData.append('studyLink', studyLink);
        formData.append('modelName', modelName);
        formData.append('description', description);
        formData.append('resolution', resolution);
        formData.append('dataset', dataset);
    
        fetch('http://localhost:5000/upload', {
          method: 'POST',
          body: formData,
          credentials: 'include',
        })
        .then(response => {
          if (!response.ok) {
            return response.json().then(errorData => {
              throw new Error(errorData.message || 'File upload failed');
            });
          }
          return response.json();
        })
        .then(formData => {
          if (formData.status === 'success') {
            console.log('File upload success');
            alert('File uploaded successfully');
          } else {
            alert(`File upload failed: ${formData.message}`);
          }
        })
        .catch(error => {
          console.error('Error during file upload:', error);
          alert(`Error during file upload: ${error.message}`);
        });
      };

      const handlePlotGeneration = (e) => {
        e.preventDefault();

        // Validation of submission IDs
        if (!Number.isInteger(Number(submissionId1)) || (submissionId2 && !Number.isInteger(Number(submissionId2)))) {
            alert("Please enter valid integer IDs");
            return;
        }

        const requestData = {
            submissionId1: submissionId1,
            submissionId2: submissionId2 || null, // If submissionId2 is empty, send null
        };

        // Sending request data (submission ids) to the backend to generate plots
        fetch('http://localhost:5000/generate_plots', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify(requestData),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.message || 'Plot generation failed');
                });
            }
            return response.blob(); // Binary data, so images expected
        })
        .then(blob => {
            // Create a link to download the plot
            saveAs(blob, 'generated_plots.zip');
            alert('Plots generated and downloaded successfully');
        })
        .catch(error => {
            console.error('Error during plot generation:', error);
            alert(`Error during plot generation: ${error.message}`);
        });
    };

    return (
        <>
        <NavBar />
        <div>
            <h1>Welcome to Your Profile</h1>
            <p>Team Name: {profileData.teamname}</p>
            <p>Email: {profileData.email}</p>
            <div className="hero-container">
              <div className="HeroFormContainer"> 
              <div style={{ textAlign: "left" }}>
              <h2 className="mb-3 upload-header">Upload your results here</h2>
              <form onSubmit={handleCSVUpload} className="form-container">
                <div className="form-group">
                  <label htmlFor="csvFileInput">Upload CSV File</label>
                  <input
                    type="file"
                    id="csvFileInput"
                    accept=".csv"
                    onChange={handleOnChange}
                    required
                    className="form-control"
                  />
                </div>
  
                <div className="form-group">
                  <label htmlFor="studyLink">Link to peer-reviewed study</label>
                  <input
                    type="url"
                    className="form-control"
                    id="studyLink"
                    aria-describedby="studyLinkHelp"
                    value={studyLink}
                    onChange={e => setStudyLink(e.target.value)}
                    placeholder="https://www.example.com"
                  />
                  <small id="studyLinkHelp" className="form-text">
                    (Optional) Input the link to a peer-reviewed study detailing your model.
                  </small>
                </div>
  
                <div className="form-group">
                  <label htmlFor="modelName">Model Name</label>
                  <input
                    type="text"
                    id="modelName"
                    value={modelName}
                    onChange={e => setModelName(e.target.value)}
                    required
                    className="form-control"
                  />
                </div>
  
                <div className="form-group">
                  <label htmlFor="description">Description</label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={e => setDescription(e.target.value)}
                    className="form-control"
                  />
                </div>
                <div className="form-group">
                    <label htmlFor="resolution">Resolution</label>
                    <select
                      id="resolution"
                      value={resolution}
                      onChange={(e) => setResolution(e.target.value)}
                      className="form-control"
                    >
                    <option value="28x28">28x28</option>
                    <option value="64x64">64x64</option>
                    <option value="128x128">128x128</option>
                    <option value="224x224">224x224</option>
                    </select>
                    <small className="form-text text-muted">
                     Choose the resolution the model was trained on: 28x28, 64x64, 128x128, 224x224
                    </small>
                      </div>
                                <div className="form-group">
                                    <label htmlFor="dataset">Dataset</label>
                                    <select
                                        id="dataset"
                                        value={dataset}
                                        onChange={(e) => setDataset(e.target.value)}
                                        className="form-control"
                                    >
                                        <option value="MedMNIST+">MedMNIST+</option>
                                        <option value="MedMNIST-C">MedMNIST-C</option>
                                    </select>
                                    <small className="form-text text-muted">
                                        Choose the dataset the model was evaluated on
                                    </small>
                                </div>
  
  
                <button type="submit" className="btn-submit">Upload</button>
              </form>
            </div>
            </div>
              <div className="hero-text">
                <h2>Generate Plots</h2>
                <form onSubmit={handlePlotGeneration} className="id-form">
                    <p className="form-description">
                        You can generate a few plots visualizing the results of your last uploaded submission.
                        Enter the id of the submission that you want to compare your results with.
                    </p>
                    <div className="form-group">
                        <label htmlFor="submissionId1">Enter First Submission ID</label>
                        <input
                            type="number"
                            id="submissionId1"
                            value={submissionId1}
                            onChange={(e) => setSubmissionId1(e.target.value)}
                            className="form-control"
                            placeholder="e.g., 123"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="submissionId2">Enter Second Submission ID (Optional)</label>
                        <input
                            type="number"
                            id="submissionId2"
                            value={submissionId2}
                            onChange={(e) => setSubmissionId2(e.target.value)}
                            className="form-control"
                            placeholder="e.g., 456"
                        />
                    </div>
                    <button type="submit" className="btn-submit">Generate Plots</button>
                </form>
             </div>
            </div>
            {/* Display a user's submissions */}
            <div className="submissions-container">
                    <h2>Your Submissions</h2>
                    {submissions.length > 0 ? (
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>CSV File ID</th>
                                    <th>Upload Date</th>
                                    <th>Resolution</th>
                                    <th>Model Name</th>
                                    <th>Test AUC</th>
                                    <th>Test Balanced Accuracy</th>
                                    <th>Test Co</th>
                                    <th>Dataset</th>
                                </tr>
                            </thead>
                            <tbody>
                                {submissions.map((submission) => (
                                    <tr key={submission.csv_file_id}>
                                        <td>{submission.csv_file_id}</td>
                                        <td>{new Date(submission.upload_date).toLocaleString()}</td>
                                        <td>{submission.resolution}</td>
                                        <td>{submission.model_name}</td>
                                        <td>{submission.test_auc}</td>
                                        <td>{submission.test_balacc}</td>
                                        <td>{submission.test_co}</td>
                                        <td>{submission.dataset}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <p>You have no submissions yet.</p>
                    )}
                </div>
        </div>
        </>
    );
}

export default Profile;