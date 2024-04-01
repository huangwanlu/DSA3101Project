import React, { useState } from 'react';

function DataManagementComponent() {
    const [file, setFile] = useState(null);

    const handleUploadClick = () => {
        if (file) {
            console.log('File uploaded:', file.name);
            // Add your file upload logic here
            alert('File uploaded successfully!');
        } else {
            console.log('No file selected for upload.');
            alert('Please select a CSV file to upload.');
        }
    };

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleExportClick = () => {
        console.log('Export button clicked. Implement export functionality here.');
        alert('Export functionality to be implemented.');
        // Add your export logic here
    };

    return (
        <div id="app">
            <h1>Data Upload and Export</h1>
            <input type="file" id="csvUpload" accept=".csv" onChange={handleFileChange} />
            <button id="uploadBtn" onClick={handleUploadClick}>Upload CSV File</button>
            <button id="exportBtn" onClick={handleExportClick}>Export Data/Visualizations</button>

            <div className="instructions">
                <h3>Instructions for Use:</h3>
                <ul>
                    <li>Please upload your dataset in CSV format only.</li>
                    <li>Click the "Upload CSV File" button after selecting your file.</li>
                    <li>Ensure your CSV file is properly formatted to avoid upload errors.</li>
                    <li>To export data, click the "Export Data/Visualizations" button once your data is ready.</li>
                </ul>
            </div>
        </div>
    );
}

export default DataManagementComponent;