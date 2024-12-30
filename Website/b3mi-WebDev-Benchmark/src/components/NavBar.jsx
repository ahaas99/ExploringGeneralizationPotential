import { useEffect, useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import "./NavBar.css";

function NavBar() {

  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const location = useLocation(); // Get the current location
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is logged in by fetching session data from the backend
    fetch('http://localhost:5000/profile', {
      method: 'GET',
      credentials: 'include',
    })
    .then(response => response.json())
    .then(data => {
      if (data.status !== "failed") {
        setIsLoggedIn(true);
      } else {
        setIsLoggedIn(false);
      }
    });
  }, [location]);

  const handleLogout = () => {
    fetch('http://localhost:5000/logout', {
      method: 'GET',
      credentials: 'include',
    })
    .then(response => response.json()) 
    .then(data => {
      if (data.status === "success") {
        setIsLoggedIn(false);
        navigate('/login');
      }
    });
  }

  return (
    <Navbar expand="lg" className="NavBar">
      <Container className = "margeExp">
      <Navbar.Brand as={Link} to="/home" className="whiteFont" id="NavBarStar">
          BIG Benchmark
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="me-auto whiteFont">
            <Nav.Link as={Link} to="/home" className="whiteFont mt-2">Home</Nav.Link>
            <Nav.Link as={Link} to="/dataset" className="whiteFont mt-2">Datasets</Nav.Link>
            <Nav.Link as={Link} to="/challenge" className="whiteFont mt-2">Challenge</Nav.Link>
            <Nav.Link as={Link} to="/leaderboard" className="whiteFont mt-2">Leaderboard</Nav.Link>
            <Nav.Link as={Link} to="/research" className="whiteFont mt-2">Research</Nav.Link>
            <Nav.Link as={Link} to="/about" className="whiteFont mt-2">About Us</Nav.Link>
            <Nav.Link as={Link} to="/profile" className="whiteFont mt-2">Profile</Nav.Link>
          </Nav>
        </Navbar.Collapse>
        <div>
        {!isLoggedIn ? (
            <>
              <Link to="/login" className="navbar-button mt-2">Login</Link>
              <Link to="/register" className="navbar-button mt-2">Register</Link>
            </>
          ) : (
            <button className="navbar-button mt-2" onClick={handleLogout}>Logout</button>
          )}
          </div>
      </Container>
    </Navbar>
  );
}

export default NavBar;