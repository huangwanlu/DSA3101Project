import React from 'react';
import './ExportReport.css'; 
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas'; 

function DataExport() {
    const handleExportClick = async () => {
        const input = document.getElementById('dashboard'); // Ensure your dashboard's root element has this ID
        const canvas = await html2canvas(input);
        const imgData = canvas.toDataURL('image/png');
        
        const pdf = new jsPDF({
            orientation: 'landscape',
        });
        
        const imgProps= pdf.getImageProperties(imgData);
        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
        pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
        pdf.save('dashboard-report.pdf');
    };

    return (
        <div>
            <button id="exportBtn" className="download-button" onClick={handleExportClick}>
                <span className="icon-download"></span> {/* span for icon */}
                Download Report
            </button>
        </div>
    );
}

export default DataExport