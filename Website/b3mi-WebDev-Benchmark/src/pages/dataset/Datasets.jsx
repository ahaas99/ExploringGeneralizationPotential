import React, { useState, useEffect } from 'react';
import NavBar from "../../components/NavBar";
import Hero from "./DataHero";
import Accordion from "./Accordion";
import HeroCorrupted from "./DataHeroCorrupted";

function Datasets() {
    const [dataset, setDataset] = useState('MedMNIST+');
    const datasets = ['MedMNIST+', 'MedMNIST-C'];
    
    if (dataset === 'MedMNIST+') {
        return (
            <>
                <NavBar />
                <div className="container mt-4 d-flex justify-content-center align-items-center">
                    <div className="row mb-4 w-100">
                        <div className="col-md-6 mx-auto">
                            <label htmlFor="resolutionSelect" className="form-label">Dataset:</label>
                            <select
                                id="resolutionSelect"
                                className="form-select"
                                value={dataset}
                                onChange={(e) => setDataset(e.target.value)}
                            >
                                {datasets.map((res) => (
                                    <option key={res} value={res}>{res}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>
                <Hero />
                <Accordion />
            </>
        );
    } else {
        return (
            <>
                <NavBar />
                <div className="container mt-4 d-flex justify-content-center align-items-center">
                    <div className="row mb-4 w-100">
                        <div className="col-md-6 mx-auto">
                            <label htmlFor="resolutionSelect" className="form-label">Dataset:</label>
                            <select
                                id="resolutionSelect"
                                className="form-select"
                                value={dataset}
                                onChange={(e) => setDataset(e.target.value)}
                            >
                                {datasets.map((res) => (
                                    <option key={res} value={res}>{res}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>
                <HeroCorrupted />
                <Accordion />
            </>
        );
    }

}


export default Datasets;