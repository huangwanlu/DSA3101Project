import React from 'react';
import { Box, IconButton, useTheme } from "@mui/material";
import { useContext } from "react";
import { ColorModeContext } from "../../theme"; //function we defined in theme.js

// import the icons for search, light&dark mode
import LightModeIcon from '@mui/icons-material/LightMode';
import NightlightIcon from '@mui/icons-material/Nightlight';

const Topbar = () => {
  const theme = useTheme(); 
  const colorMode = useContext(ColorModeContext); //allow us to toggle between different states

  return (
   <Box display="flex" justifyContent="space-between" p={2}> 

      {/* SEARCH BAR & ICONS for light&dark mode*/}
      <Box display="flex" alignItems="center">
        <IconButton onClick={colorMode.toggleColorMode}>
          {theme.palette.mode === "light" ? (
            <LightModeIcon fontSize="small" />
          ) : (
            <NightlightIcon fontSize="small" />
          )}
        </IconButton>

      </Box>
  
   </Box> 
  );
};

export default Topbar;