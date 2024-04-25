import React, { useState } from 'react';
import { Box, Typography, useTheme } from "@mui/material";
import { tokens } from "./theme";
import Header from "./components/Header";
import PersonOffOutlinedIcon from '@mui/icons-material/PersonOffOutlined';
import DataExport from './ExportReport';
import PieChart from "./components/PieChart";
import BarChart from "./components/BarChart";
import StatBox from "./components/Statbox";
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import PieChart2 from "./components/PieChart2";

const Dashboard = ({ dashboardRef }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  
  return (
    <DataExport>
      <Box m="20px">
        {/* HEADER */}
        <Box display="flex" justifyContent="space-between" flexDirection="column" alignItems="flex-start">
          <Header title="Customer Churn Data" subtitle="" />

          {/* New Subtitle Below Header with Custom Color */}
          <Typography
            variant="subtitle1"
            style={{ color: "#8399e6", marginTop: -35, paddingTop: -40, fontWeight: 3000, fontSize: '1rem' }} // This sets the subtitle color
            mt={2}
          >
            The churn analysis is based on the dataset titled “Bank Customer Churn Dataset” from Kaggle, which contains information on account holders from ABC Multistate Bank. The primary objective is to predict customer churn, a critical factor for banks aiming to sustain their business by retaining customers.
          </Typography>
        </Box>

        {/* GRID & CHARTS */}
        <Box
          display="grid"
          gridTemplateColumns="repeat(12, 1fr)"
          gridAutoRows="140px"
          gap="20px"
        >
          {/* ROW 1 */}
          <Box
              gridColumn="span 2"
              backgroundColor={colors.black[400]}
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
               <StatBox
                title="Overall Churn Rate"
                subtitle={
                    <Typography style={{ color: "#8399e6", fontSize: '2rem',fontWeight: 'bold'  }}>20.37%</Typography> // Replace #desiredColor with the color you want.
                  }
                icon={
                  <PersonOffOutlinedIcon
                    sx={{ color: colors.green[600], fontSize: "60px" }}
                  />
                }
              />
            </Box>
            <Box
  gridColumn="span 5"
  backgroundColor={colors.black[400]}
  display="flex"
  flexDirection="column"
  alignItems="center"
  justifyContent="center"
  textAlign="center" // Center align text
  padding="20px" // Add padding to the box
>
  <Typography
    variant="h3"
    fontWeight="800"
    color="#8399e6"
    mt="0" // Remove negative margin
    mb="0px" // Add bottom margin for spacing
  >
    Churned Customers by Gender
  </Typography>
  <PieChart />
</Box>
<Box
  gridColumn="span 5"
  backgroundColor={colors.black[400]}
  display="flex"
  flexDirection="column"
  alignItems="center"
  justifyContent="center"
  textAlign="center" // Center align text
  padding="20px" // Add padding to the box
>
  <Typography
    variant="h3"
    fontWeight="800"
    color="#8399e6"
    mt="0" // Remove negative margin
    mb="0px" // Add bottom margin for spacing
  >
    Churned Customers by Activeness
  </Typography>
  <PieChart2 />
</Box>

          {/* ROW 2 */}
          <Box
            gridColumn="span 6"
            gridRow="span 2" //make the box longer
            backgroundColor={colors.black[400]}
          >
            <Box
              mt="25px"
              p="0 30px"
              display="flex "
              justifyContent="space-between"
              alignItems="center"
            >
              <Typography
                variant="h3"
                fontWeight="1000"
                color="#white"
                mt="-20px"
              >
                Ranked Churn Drivers
              </Typography>

              <List component="nav" aria-label="Feature scores" style={{ lineHeight: '1.2', margin: 0, padding: 0 }} >
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>1. Age</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>2. Credit Score</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>3. Estimated Salary</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>4. Balance</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>5. Products Number</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>6. Tenure</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>7. Active Member</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>8. Country</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>9. Credit Card</ListItem>
                <ListItem style={{ fontWeight: "1000", color: "#8399e6", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.1rem' }}>10. Gender</ListItem>
              </List>
            </Box>
          </Box>
          <Box
            gridColumn="span 6"
            gridRow="span 2"
            backgroundColor={colors.black[400]}
            overflow="auto"
          >

              <BarChart />
            </Box>
          </Box>
        </Box>

    </DataExport>
  );
}; 

export default Dashboard;