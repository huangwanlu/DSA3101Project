import React from 'react';
import { Box, Typography, useTheme } from "@mui/material";
import { tokens } from "../theme";

const StatBox = ({ title, subtitle, icon }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  return (
    <Box width="100%" textAlign="center"> {/* Center aligning */}
      <Box display="flex" justifyContent="center" alignItems="center" flexDirection="column"> {/* Center aligning */}
        <Box mb={1}> {/* Adding margin bottom */}
          {icon}
          <Typography
            variant="h4"
            fontWeight="bold"
            sx={{ color: colors.grey[100] }}
          >
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" sx={{ color: "#525dfa" }}>
          {subtitle}
        </Typography>
      </Box>
    </Box>
  );
};

export default StatBox;