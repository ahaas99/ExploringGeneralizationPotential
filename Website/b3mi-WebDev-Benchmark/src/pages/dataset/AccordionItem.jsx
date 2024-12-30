import { useEffect, useRef, useState } from "react";
import "./AccordionItem.css";
import { Collapse } from "bootstrap";
import PropTypes from 'prop-types';

const AccordionItem = ({ id, Title, Body }) => {
  const collapseRef = useRef(null); // Correctly initialize useRef
  const [isExpanded, setIsExpanded] = useState(true); // Used to control icon rotation

  useEffect(() => {
    const collapseElement = collapseRef.current;
    const bsCollapse = new Collapse(collapseElement, {
      toggle: false // Prevent automatic toggling on initialization
    });

    // Ensure the collapse is open if isExpanded is true
    if (isExpanded) {
      bsCollapse.show();
    }

    const onShow = () => {
      setIsExpanded(true); // Set expanded true when collapse is opening
    };
    const onHide = () => {
      setIsExpanded(false); // Set expanded false when collapse is closing
    };

    // Add event listeners for collapse show and hide
    collapseElement.addEventListener("show.bs.collapse", onShow);
    collapseElement.addEventListener("hide.bs.collapse", onHide);

    // Cleanup function to remove event listeners
    return () => {
      collapseElement.removeEventListener("show.bs.collapse", onShow);
      collapseElement.removeEventListener("hide.bs.collapse", onHide);
    };
  }, []); // Empty dependency array ensures effect runs only once after initial render

  return (
    <>
      <div className="accordion-item marginContainers">
        <h2 className="accordion-header">
          <button
            className="accordion-button testColour"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target={`#panelsStayOpen-collapseOne${id}`}
          >
            {Title}
            <i className={`bi bi-caret-up-fill ${isExpanded ? '' : 'arrow-rotate'}`}></i>
          </button>
        </h2>
        <div
          id={`panelsStayOpen-collapseOne${id}`}
          className={`accordion-collapse collapse ${isExpanded ? 'show' : ''}`}
          ref={collapseRef}
        >
          <div className="accordion-body">{Body}</div>
        </div>
      </div>
    </>
  );
};

// Define prop types for AccordionItem component
AccordionItem.propTypes = {
  id: PropTypes.string.isRequired,  // id is a string and is required
  Title: PropTypes.string.isRequired,  // Title is a string and is required
  Body: PropTypes.element.isRequired  // Body is a React component (React element) and is required
};

export default AccordionItem;
