import React, { useState, useEffect } from "react";
import { Box, Typography, Button, useTheme } from "@mui/material"; 
import { DataGrid } from "@mui/x-data-grid";
import { tokens } from "../../theme";
import Header from "../../components/Header"; 
import './Churn.css';
import { FileDownloadOutlined as FileDownloadOutlinedIcon } from '@mui/icons-material';


const Churn = () => {
    const theme = useTheme();
    const colors = tokens(theme.palette.mode);
    const columns = [
    { field: "customer_id", headerName: "Customer ID" },
    {
      field: "credit_score",
      headerName: "Credit Score",
      type: "number",
      headerAlign: "left", 
      align: "left",
    },
    {
      field: "country",
      headerName: "Country",
    },
    {
      field: "gender",
      headerName: "Gender",
    },
    {
      field: "age",
      headerName: "Age",
      type: "number",
      headerAlign: "left", 
      align: "left",
    },
    {
      field: "tenure",
      headerName: "Tenure",
      type: "number",
      headerAlign: "left",
      align: "left",
    },
    {
        field: "balance",
        headerName: "Balance",
        type: "number",
        headerAlign: "left",
        align: "left",
    },
    {
        field: "products_number",
        headerName: "Product Number",
        type: "number",
        headerAlign: "left",
        align: "left",
    },
    {
        field: "credit_card",
        headerName: "Has Credit Card?",
        type: "number",
        headerAlign: "left",
        align: "left",
    },
    {
        field: "active_member",
        headerName: "Is Active Member?",
        type: "number",
        headerAlign: "left",
        align: "left",
    },
    {
        field: "estimated_salary",
        headerName: "Estimated Salay",
        type: "number",
        headerAlign: "left",
        align: "left",
    },
    {
        field: "churn",
        headerName: "Churn",
        type: "number",
        headerAlign: "left",
        align: "left",
    },
  ];

    const [churnData, setChurnData] = useState([]);
    const [dataFetched, setDataFetched] = useState(false); // New state variable to track data fetching status

    const [file, setFile] = useState(null);
    const [showTooltip, setShowTooltip] = useState(false);

    console.log("Tooltip visibility state:", showTooltip);
    
    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUploadClick = async (event) => {
        event.preventDefault();

        const formData = new FormData();
        formData.append('file_upload', file);

        try {
            const endpoint = "http://localhost:8000/uploadfile/"
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            setChurnData(data);
            setDataFetched(true);
            console.log(churnData)


            if (response.ok) {
                console.log("File uploaded successfully!")
            } else {
                console.error("Failed to upload file.");
            }
        } catch (error) {
            console.error(error);
        }
    }

    const iconStyle = {
        marginRight: '8px',
    };

    // CSV Export Handler
    const handleExport = () => {
        const csvData = churnData.map(row => 
            columns.map(column => `"${row[column.field] || ''}"`).join(',')
        );
        const csvContent = [
            columns.map(column => column.headerName).join(','), // header row
            ...csvData // data rows
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', 'churn_data.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    return (
        <Box m="20px">
            <Header title="Data" subtitle="Customer Churn Data" />
            <div className="data-upload-container">
                <div className="file-inputs">
                    <input type="file" id="csvUpload" accept=".csv" onChange={handleFileChange} />
                    <button id="uploadBtn" onClick={handleUploadClick}>Upload CSV File</button>
                    <span 
                        className="info-tooltip" 
                        onMouseEnter={() => setShowTooltip(true)}
                        onMouseLeave={() => setShowTooltip(false)}
                    >
                        <span className="question-mark">?</span>
                        {showTooltip && (
                            <div className="tooltip-content">
                                <p>Instructions for Use:</p>
                                <p>Please upload your dataset in CSV format only.</p>
                                <p>Click the "Upload CSV File" button after selecting your file.</p>
                                <p>Columns Required:</p>
                                <ul>
                                    <li>customer_id: int</li>
                                    <li>credit_score: int</li>
                                    <li>country: string ***</li>
                                    <li>gender: Female/Male</li>
                                    <li>age: int</li>
                                    <li>tenure: int</li>
                                    <li>balance: int</li>
                                    <li>products_number: int from 1-4</li>
                                    <li>credit_card: 1 for yes, 0 for no</li>
                                    <li>active_member: 1 for yes, 0 for no</li>
                                    <li>estimated_salary: int</li>
                                </ul>
                            </div>
                        )}
                    </span>
                </div>
                {dataFetched && (
                    <Button
                    variant="contained"
                    onClick={handleExport}
                    className="export-button"
                    sx={{
                      backgroundColor: '#3f51b5', // Your desired color
                      color: 'white',
                      fontFamily: 'Arial', // Replace with the font-family used by other buttons
                    fontSize: '0.875rem', // Replace with the font-size used by other buttons, if necessary
                    fontWeight: 'bold',
                      //marginLeft: 'auto', 
                      marginRight: '10px'
                    }}
                  >
                    <FileDownloadOutlinedIcon style={iconStyle} />
                    Export to CSV
                  </Button>
                )}
            </div>
            {dataFetched ? (
                <DataGrid 
                    rows={churnData} 
                    columns={columns} 
                    getRowId={(row) => row.customer_id + row.products_number}
                    pageSize={5}
                    rowsPerPageOptions={[5]}
                />
            ) : (
                <Typography>Loading...</Typography>
            )}
        </Box>
    );
};

export default Churn;