import { BrowserRouter, Routes, Route } from "react-router-dom";
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
import "./App.css";
import Home from "./pages/index/Home";
import Dataset from "./pages/dataset/Datasets";
import Leaderboard from "./pages/leaderboard/LeaderboardPage";
import Challenge from "./pages/challenge/ChallengePage";
import EvaluationMetrics from "./pages/metrics/EvaluationMetrics";
import Research from "./pages/research/ResearchPage";
import AboutUs from "./pages/about/AboutUs";
import Register from "./pages/register/RegisterPage"
import Login from "./pages/login/LoginPage"
import NoPage from "./pages/nopage/NoPage404";
import Profile from './pages/profile/UserProfile';

function App() {
  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route index element={<Home />} />
          <Route path="/home" element={<Home />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/challenge" element={<Challenge />} />
          <Route path="/metrics" element={<EvaluationMetrics />} />
          <Route path="/research" element={<Research />} />
          <Route path="/about" element={<AboutUs />} />
          <Route path="/register" element={<Register />} />
          <Route path="/login" element={<Login />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="*" element={<NoPage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
