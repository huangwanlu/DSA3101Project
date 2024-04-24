import React, { useRef } from 'react';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas'; 
import DownloadButton from './DownloadButton';

function DataExport({ children }) {
    const dashboardRef = useRef(null); // Declare dashboardRef here
    
    const handleExportClick = async () => {
        if (dashboardRef.current) {
          const canvas = await html2canvas(dashboardRef.current);
          const imgData = canvas.toDataURL('image/png');
          
          const pdf = new jsPDF({
            orientation: 'landscape',
          });
          
          const imgProps = pdf.getImageProperties(imgData);
          const pdfWidth = pdf.internal.pageSize.getWidth();
          const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
          pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
          pdf.save('dashboard-report.pdf');
        } else {
          console.error('Dashboard element not found or not yet rendered in the DOM');
        }
      };

      return (
        <div>
            <div ref={dashboardRef}>
                {children} {/* Render children, which is the Dashboard content */}
            </div>
            <DownloadButton handleExportClick={handleExportClick} />
        </div>
    );
}

export default DataExport;