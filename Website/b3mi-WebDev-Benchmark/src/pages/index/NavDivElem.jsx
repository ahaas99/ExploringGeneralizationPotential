import './NavDivElem.css';
import PropTypes from 'prop-types';

const NavDivElem = ({ icon, header, description, link, linkText }) => {

    const handleClick = () => {
        // Navigate to the link
        window.location.href = link;
      };

  return (
    <div className="ContainerNavDivElem" onClick={handleClick}>
      <div>{icon}</div>
      <h2>{header}</h2>
      <p className="NavElemParagraph">{description}</p>
      <div className="linkWrapper">
        <a href={link}>{linkText}</a>
        <p className="linkAppendage">&#62;</p>
      </div>
      
    </div>
  );
};

NavDivElem.propTypes = {
  icon: PropTypes.element.isRequired,  // id is a string and is required
  header: PropTypes.string.isRequired,  // Title is a string and is required
  description: PropTypes.string.isRequired,  // Body is a React component (React element) and is required
  link: PropTypes.string.isRequired,
  linkText: PropTypes.string.isRequired
};


export default NavDivElem;
