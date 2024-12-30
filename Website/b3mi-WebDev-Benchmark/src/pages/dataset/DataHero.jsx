import "./DataHero.css";
import Image from 'react-bootstrap/Image';
import medMnistImage from '../../assets/MedMNIST-v2.png';

const Hero = () => {
  return (
    <>
      <div className="data-hero-container">
      <h1 id="HeroHeader">MedMNIST+</h1>
      <div id="medMNISTIMG">
        <Image
          src={medMnistImage}
          className="img-fluid"
          alt="Overview of the datasets included in MedMNIST+"/>
      </div>
        {/* Container for entire Hero*/}
        <div>
          <p id="HeroText">
            The BAM! Benchmark is based on the MedMNIST+ dataset, a collection of 12 biomedical imaging datasets in 2D,
            available in the resolutions 28x28, 64x64, 128x128 and 224x224. MedMNIST+ represents a diverse and standardized compilation of
            datasets, suited for testing model performance not just on the wide-ranging medical domains but also on datasets with different resolutions.
            The classification tasks include binary/multi-class classification, multi-label classification as well as ordinal regression.
          </p>
          <a href={"https://zenodo.org/records/10519652"} className="hero-button">
            {"Get Dataset"}
          </a>
          <a href={"https://www.nature.com/articles/s41597-022-01721-8"} className="hero-button">
            {" "}
            {"Paper"}
          </a>
        </div>
      </div>
    </>
  );
};

export default Hero;
