import "./DatasetDescription.css";
import "bootstrap-icons/font/bootstrap-icons.css";
import PropTypes from 'prop-types';

const DatasetDescription = ({
  image,
  domain,
  classificationTask,
  numImages,
  numClasses,
  description,
}) => {
  return (
    <div className="container text-center">
      <div className="row">
        <div className="col colHeight">
          <img
            src={image}
            className="img-fluid imageFormatting"
            alt="Overview of the datasets included in MedMNIST+"
          />
        </div>
        <div className="col textMargin">
          <div className="row justifyText">{description}</div>
        </div>
        <div className="col">
          <div className="row centerContents">
            <div className="flexContainer">
              <i className="bi bi-play-fill iconColour"></i>
              <pre>&ensp;</pre>
              Domain: {domain}
            </div>
          </div>
          <div className="row centerContents">
            <div className="flexContainer">
              <i className="bi bi-play-fill iconColour"></i>
              <pre>&ensp;</pre>
              Classification Task: {classificationTask}
            </div>
          </div>
        </div>
        <div className="col">
          <div className="row centerContents">
            <div className="flexContainer">
              <i className="bi bi-play-fill iconColour"></i>
              <pre>&ensp;</pre>
              Number of Images: {numImages}
            </div>
          </div>
          <div className="row centerContents">
            <div className="flexContainer">
              <i className="bi bi-play-fill iconColour"></i>
              <pre>&ensp;</pre>
              {numClasses}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Define prop types for AccordionItem component
DatasetDescription.propTypes = {
  image: PropTypes.string.isRequired,  // id is a string and is required
  domain: PropTypes.string.isRequired,  // Title is a string and is required
  classificationTask: PropTypes.string.isRequired,  // Body is a React component (React element) and is required
  numImages: PropTypes.string.isRequired,
  numClasses: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired
};

export default DatasetDescription;
