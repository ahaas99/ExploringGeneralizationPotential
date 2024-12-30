import "./DataHero.css";
import Image from 'react-bootstrap/Image';
import medMnistImage from '../../assets/MedMNISTC.png';

const HeroCorrupted = () => {
  return (
    <>
      <div className="data-hero-container">
      <h1 id="HeroHeader">MedMNIST-C</h1>
      <div id="medMNISTIMG">
        <Image
          src={medMnistImage}
          className="img-fluid"
          alt="Overview of the datasets included in MedMNIST+"/>
      </div>
        {/* Container for entire Hero*/}
        <div>
          <p id="HeroText">
                      The BAM! Benchmark introduces also a challenge to test the robustness to distortions of presented models.
                      The challenge is based on the MedMNIST-C databse. A corrupted version of the 12 2D datasets of the MedMNIST+ collection.
                      The corruptions are specifically designed for each dataset to mimic the types of artifacts that may arise during image acquisition and processing, spanning five severity levels.
                      This simulates real-world anomalies or potential shifts in data distribution. The dataset and code for 224x224 is available from the original paper.
                      Since not all corruptions are suitable for the lower resolution images the code needs to be adapted for every other resolution. 
                  </p>
          <a href={"https://github.com/francescodisalvo05/medmnistc-api"} className="hero-button">
                      {"Get Dataset for 224x224"}
          </a>
          <a href={"https://github.com/francescodisalvo05/medmnistc-api"} className="hero-button">
                          {"Get Dataset"}
          </a>
                  
          <a href={"https://arxiv.org/abs/2406.17536"} className="hero-button">
            {" "}
            {"Paper"}
          </a>
        </div>
      </div>
    </>
  );
};

export default HeroCorrupted;
