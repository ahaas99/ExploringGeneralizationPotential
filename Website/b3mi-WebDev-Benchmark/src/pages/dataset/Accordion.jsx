import AccordionItem from "./AccordionItem";
import DatasetDescription from "./DatasetDescription";
import pathMNIST from "../../assets/PathMNIST.png";
import chestMNIST from "../../assets/ChestMNIST.png";
import dermaMNIST from "../../assets/DermaMNIST.png";
import octMNIST from "../../assets/OctMNIST.png";
import pneumoniaMNIST from "../../assets/PneumoniaMINST.png";
import retinaMNIST from "../../assets/RetinaMNIST.png";
import breastMNIST from "../../assets/BreastMNIST.png";
import bloodMNIST from "../../assets/BloodMNIST.png";
import tissueMNIST from "../../assets/TissueMNIST.png";
import organAMNIST from "../../assets/OrganAMNIST.png";
import organCMNIST from "../../assets/OrganCMNIST.png";
import organSMNIST from "../../assets/OrganSMNIST.png";

const Accordion = () => {
  return (
    <>
      <div className="accordion" id="accordionExample">
        <AccordionItem
          id="PathMNIST"
          Title="PathMNIST"
          Body={
            <DatasetDescription
              image={pathMNIST}
              domain="Colorectal cancer"
              classificationTask="multi-class"
              numImages="107,180"
              numClasses="Number of Classes: 9"
              description="PathMNIST encompasses some 107,180 images of colorectal cancer histology slides. Samples of tissue were obtained from hematoxylin–eosin–stained tissue slides and hand-delineated into non-overlapping image patches. "
            />
          }
        />
        <AccordionItem
          id="ChestMNIST"
          Title="ChestMNIST"
          Body={
            <DatasetDescription
              image={chestMNIST}
              domain="Chest X-Ray"
              classificationTask="multi-label binary class"
              numImages="112,120"
              numClasses="Number of Classes: 14"
              description="ChestMNIST builds on the NIH-ChestXray14 dataset, consisting of images of chest X-rays to identify thoracic diseases. Frontal X-Ray images were taken of some 30,805 patients."
            />
          }
        />
        <AccordionItem
          id="DermaMNIST"
          Title="DermaMNIST"
          Body={
            <DatasetDescription
              image={dermaMNIST}
              domain="Dermatoscopy"
              classificationTask="multi-class"
              numImages="10,015"
              numClasses="Number of Classes: 7"
              description="DermaMNIST uses the 10,015 dermatoscopic images from the HAM10000 Challenge. Originating from 2 different institutions and having been acquired over the course of 20 years, the images were acquired using different imaging methods."
            />
          }
        />
        <AccordionItem
          id="OctMNIST"
          Title="OctMNIST"
          Body={
            <DatasetDescription
              image={octMNIST}
              domain="Optical Coherence Tomography"
              classificationTask="multi-class"
              numImages="109,309"
              numClasses="Number of Classes: 4"
              description="OctMNIST comprises optical coherence tomography images of the retina pertaining to diagnoses of age-related macular degeneration as well as diabetic macular edema. 5 institutions shared their data collected from 2013-2017."
            />
          }
        />
        <AccordionItem
          id="PneumoniaMNIST"
          Title="PneumoniaMNIST"
          Body={
            <DatasetDescription
              image={pneumoniaMNIST}
              domain="Pneunomia"
              classificationTask="binary-class"
              numImages="5,856"
              numClasses="Number of Classes: 2"
              description="PneumoniaMNIST contains 5,856 pediatric chest x-ray images for binary classification. Originally used to examine generalizability of a model previously trained on OctMNIST, these radiographic images add another medical domain to MedMNIST+."
            />
          }
        />
        <AccordionItem
          id="RetinaMNIST"
          Title="RetinaMNIST"
          Body={
            <DatasetDescription
              image={retinaMNIST}
              domain="Pneunomia"
              classificationTask="ordinal regression"
              numImages="1,600"
              numClasses="Number of Levels: 5"
              description="RetinaMNIST offers the chance to perform ordinal regression and distinguishes into 5 clinically defined labels (levels) of diabetic retinopathy. It contains 1,600 retinal fundus images from the DeepDRiD challenge."
            />
          }
        />
        <AccordionItem
          id="BreastMNIST"
          Title="BreastMNIST"
          Body={
            <DatasetDescription
              image={breastMNIST}
              domain="Breast Cancer"
              classificationTask="binary-class"
              numImages="780"
              numClasses="Number of Classes: 2"
              description="BreastMNIST revolves around classifying ultrasound scans to show malignant cancer or not. The data comes from over 600 women and was collected at the Baheya Hospital for Early Detection & Treatment of Women's Cancer in Cairo, Egypt."
            />
          }
        />
        <AccordionItem
          id="BloodMNIST"
          Title="BloodMNIST"
          Body={
            <DatasetDescription
              image={bloodMNIST}
              domain="Hematology"
              classificationTask="multi-class"
              numImages="17,092"
              numClasses="Number of Classes: 8"
              description="BloodMNIST is made up of 17,092 microscopic peripheral blood cell images. Individuals which contributed to this dataset did not have any sort of infection, hematologic or oncologic disease and did not undergo pharmacologic treatment."
            />
          }
        />
        <AccordionItem
          id="TissueMNIST"
          Title="TissueMNIST"
          Body={
            <DatasetDescription
              image={tissueMNIST}
              domain="Kidney Tissue"
              classificationTask="multi-class"
              numImages="236,386"
              numClasses="Number of Classes: 8"
              description="TissueMNIST deals with classifying certain cell types in human kidney tissue. Kidney nephrectomy tissues of 3 deceased donors to acquire the data."
            />
          }
        />
        <AccordionItem
          id="OrganAMNIST"
          Title="OrganAMNIST"
          Body={
            <DatasetDescription
              image={organAMNIST}
              domain="Liver Tumor"
              classificationTask="multi-class"
              numImages="58,850"
              numClasses="Number of Classes: 11"
              description="OrganAMNIST is part of the OrganMNIST dataset family and contains Computed Tomography (CT) images of possible liver tumors in the axial plane. The data has been collected by seven different institutions."
            />
          }
        />
        <AccordionItem
          id="OrganCMNIST"
          Title="OrganCMNIST"
          Body={
            <DatasetDescription
              image={organCMNIST}
              domain="Liver Tumor"
              classificationTask="multi-class"
              numImages="23,660"
              numClasses="Number of Classes: 11"
              description="OrganCMNIST is part of the OrganMNIST dataset family and contains Computed Tomography (CT) images of possible liver tumors in the coronal plane. The data has been collected by seven different institutions."
            />
          }
        />
        <AccordionItem
          id="OrganSMNIST"
          Title="OrganSMNIST"
          Body={
            <DatasetDescription
              image={organSMNIST}
              domain="Liver Tumor"
              classificationTask="multi-class"
              numImages="25,221"
              numClasses="Number of Classes: 11"
              description="OrganSMNIST is part of the OrganMNIST dataset family and contains Computed Tomography (CT) images of possible liver tumors in the sagittal plane. The data has been collected by seven different institutions."
            />
          }
        />
        
      </div>
    </>
  );
};

export default Accordion;
