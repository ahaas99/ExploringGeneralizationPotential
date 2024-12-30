import { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import "./RegisterForm.css";

function RegisterForm() {
    const [email, setEmail] = useState('');
    const [teamname, setTeamName] = useState('');
    const [password, setPassword] = useState('');

    const navigate = useNavigate(); // Get the navigate function
  
    const handleSubmit = (event) => {
      event.preventDefault(); // Prevent default form submission behavior
  
      // Here you can add any validation or processing before sending the data
      const formData = { email, teamname, password };
      console.log(formData);  // Debug: Log form data to ensure it's captured correctly
  
      // Example of how to send data to the server:
      fetch('http://127.0.0.1:5000/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })
      .then(response => response.json())
      .then(data => {
        console.log('Success:', data);
        navigate('/login'); // Redirect to the login page after successful registration
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    };
  return (
    <>
    
    <div className="container justify-content-center formContainer">
    <h1 id="registerHeader">Register</h1>
      <form onSubmit={handleSubmit}>
        <div className="mb-4 text-white">
          <label htmlFor="email" className="form-label">
            Email address
          </label>
          <input
            type="email"
            className="form-control"
            id="email"
            aria-describedby="emailHelp"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
          />
          <div id="emailHelp" className="form-text text-white">
            We&apos;ll never share your email with anyone else.
          </div>
        </div>
        <div className="mb-4 text-white">
          <label htmlFor="teamname" className="form-label">
            Team name
          </label>
          <input
            type="text"
            className="form-control"
            id="teamname"
            aria-describedby="emailHelp"
            value={teamname}
            onChange={e => setTeamName(e.target.value)}
            required
          />
          <div id="teamHelp" className="form-text text-white">
            Choose a unique name for your team.
          </div>
        </div>
        <div className="mb-4 text-white">
          <label htmlFor="password" className="form-label">
            Password
          </label>
          <input
            type="password"
            className="form-control"
            id="password"
            value={password} 
            onChange={e => setPassword(e.target.value)}
            required
          />
            <div id="passwordHelp" className="form-text text-white">
            Choose a strong password to protect your account.
          </div>
        </div>
        <button type="submit" className="btn btn-primary buttonColour button:hover">
          Submit
        </button>
      </form>
    </div>
    </>
  );
}

export default RegisterForm;
