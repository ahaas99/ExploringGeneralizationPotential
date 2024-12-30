import { useState } from 'react';
import "./LoginForm.css";
import { useNavigate } from 'react-router-dom';

function LoginForm() {
  const [identifier, setIdentifier] = useState('');  // Identifier for either email or teamname
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  
    const handleSubmit = (event) => {
      event.preventDefault(); // Prevent default form submission behavior
  
      const formData = { login: identifier, password };
      console.log(formData);  // For debugging
  
      // Example of how to send data to the server:
      fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Ensure cookies are included with the request
        body: JSON.stringify(formData),
      })
      .then(response => response.json())
      .then(data => {
        console.log('Success:', data);
        if (data.status === "success") {
          navigate('/profile', { state: { teamName: data.teamName } });
        } else {
          alert("Login failed!");
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    };
  return (
    <>
    <div className="container justify-content-center formContainer">
    <h1 id="loginHeader">Login</h1>
      <form onSubmit={handleSubmit}>
        <div className="mb-4 text-white">
          <label htmlFor="identifier" className="form-label">
            Email or Team name
          </label>
          <input
            type="text"
            className="form-control"
            id="identifier"
            aria-describedby="identifierHelp"
            value={identifier}
            onChange={e => setIdentifier(e.target.value)}
            required
          />
          <div id="identifierHelp" className="form-text text-white">
            Please input either your email or team name.
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
            Input your password to authenticate yourself.
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

export default LoginForm;
