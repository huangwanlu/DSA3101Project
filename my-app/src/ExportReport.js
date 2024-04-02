import React from 'react';
import './ExportReport.css';  

function DataExport() {
    const handleExportClick = () => {
        console.log('Export button clicked. Implement export functionality here.');
        alert('Export functionality to be implemented.');
        // Add export logic
    };

    return (
        <div>
            <button id="exportBtn" className="download-button" onClick={handleExportClick}>
                <span className="icon-download"></span> {/* span for icon */}
                Export Data/Visualizations
            </button>
        </div>
    );
}

export default DataExport;