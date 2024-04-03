import React, { useState } from 'react';
import './DataUpload.css'; 
import Papa from 'papaparse';

function DataUpload() {
    const [file, setFile] = useState(null);

    const handleUploadClick = () => {
        if (file) {
            Papa.parse(file, {
                complete: function(results) {
                    console.log(results.data);
                    // `results.data` is an array of rows, each row being an array of fields
                    // If CSV file includes headers, `results.data` will be an array of objects
                    alert('File uploaded and processed successfully!');
                },
                header: true, // Set to true if your CSV has headers
                skipEmptyLines: true, // Skips empty lines
                dynamicTyping: true, // Automatically converts numeric and boolean data
            });
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