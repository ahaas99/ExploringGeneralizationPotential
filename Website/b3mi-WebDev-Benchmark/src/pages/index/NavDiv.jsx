import NavDivElem from "./NavDivElem";
import "./NavDiv.css";

const NavDiv = () => {
  return (
    <div className="ContainerNavDiv">
      <div className="row">
        <div className="col">
          <NavDivElem icon={<i className="bi bi-trophy iconColour"></i>}
          header="Challenge"
          description="Get to know the exciting BIG Challenge, offering a unique chance to compete in AI research centered around generalizability in medical imaging."
          link="challenge"
          linkText="Explore Challenge">
          </NavDivElem>
        </div>
        <div className="col">          
        <NavDivElem icon={<i className="bi bi-collection iconColour" ></i>}
          header="Dataset"
          description="Get an overview of the different datasets the challenge encompasses and download them."
          link="dataset"
          linkText="Datasets Overview">
          </NavDivElem></div>
        <div className="col">
        <NavDivElem icon={<i className="bi bi-bar-chart iconColour"></i>}
          header="Leaderboard"
          description="The current leaderboard highlighting the state-of-the art performances."
          link="leaderboard"
          linkText="View Leaderboard">
          </NavDivElem>
        </div>
      </div>
      <div className="row">
        <div className="col">
        <NavDivElem icon={<i className="bi bi-clipboard-data iconColour"></i>}
          header="Evaluation Metrics"
          description="Get an insight into the evaluation metrics the BIG Benchmark uses and investigate their meaning."
          link="metrics"
          linkText="Investigate Metrics">
          </NavDivElem>
        </div>
        <div className="col">
        <NavDivElem icon={<i className="bi bi-book iconColour"></i>}
          header="Research"
          description="The research this benchmark is built upon, offering a chance to dive deeper into the exciting details this challenge encompasses."
          link="research"
          linkText="Related Works">
          </NavDivElem>
        </div>
        <div className="col">
        <NavDivElem icon={<i className="bi bi-hexagon iconColour" height="50vh"></i>}
          header="About Us"
          description="Legal Information"
          link="about"
          linkText="About Us">
          </NavDivElem>
        </div>
      </div>
    </div>
  );
};

export default NavDiv;
