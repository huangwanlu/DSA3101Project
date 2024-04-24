import React from 'react';
import { FileDownloadOutlined as FileDownloadOutlinedIcon } from '@mui/icons-material';

const DownloadButton = ({ handleExportClick }) => {
    const buttonStyle = {
        display: 'inline-flex',       // Use flexbox for inline elements
        alignItems: 'center',         // Center align vertically
        justifyContent: 'center',     // Center align horizontally
        padding: '8px 16px',          // Add some padding
        border: 'none',               // Remove the border
        borderRadius: '4px',          // Optional: rounded corners
        backgroundColor: '#3f51b5',   // Use the color from your screenshot
        color: 'white',               // Text color
        cursor: 'pointer',            // Change cursor on hover
        outline: 'none',              // Remove focus outline
        fontSize: '16px',             // Set font size
    };

    const iconStyle = {
        marginRight: '8px',
    };

    return (
        <div style={{ textAlign: 'center' }}>
            <button
                id="exportBtn"
                className="download-button"
                style={buttonStyle}
                onClick={handleExportClick}
            >
                <FileDownloadOutlinedIcon style={iconStyle} />
                Download Report
            </button>
        </div>
    );
};

export default DownloadButton;