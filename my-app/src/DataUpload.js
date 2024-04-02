import React, { useState } from 'react';

function DataUpload() {
    const [file, setFile] = useState(null);

    const handleUploadClick = () => {
        if (file) {
            console.log('File uploaded:', file.name);
            // Add file upload logic
            alert('File uploaded successfully!');
        } else {
            console.log('No file selected for upload.');
            alert('Please select a CSV file to upload.');
        }
    };

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    return (
        <div>
            <input type="file" id="csvUpload" accept=".csv" onChange={handleFileChange} />
            <button id="uploadBtn" onClick={handleUploadClick}>Upload CSV File</button>

            <div className="instructions">
                <h3>Instructions for Use:</h3>
                <ul>
                    <li>Please upload your dataset in CSV format only.</li>
                    <li>Click the "Upload CSV File" button after selecting your file.</li>
                </ul>
            </div>
        </div>
    );
}

export default DataUpload;