import { useState } from "react";
import { ProSidebar, Menu, MenuItem } from "react-pro-sidebar";
import { Box, IconButton, Typography, useTheme } from "@mui/material";
import { Link } from "react-router-dom";
import "react-pro-sidebar/dist/css/styles.css";
import { tokens } from "../../theme";


// icons
import HomeOutlinedIcon from "@mui/icons-material/HomeOutlined";
import OnlinePredictionIcon from '@mui/icons-material/OnlinePrediction';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import MenuOutlinedIcon from "@mui/icons-material/MenuOutlined";


// add in sx property to adjust size of fonts instead of using "smallerFont"
const Item = ({ title, to, icon, selected, setSelected, sx }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  return (
    <MenuItem
      active={selected === title}
      style={{
        color: colors.grey[100],
      }}
      onClick={() => setSelected(title)}
      icon={icon}
    >
      {/*<Typography variant={smallerFont ? "body2" : "body1"}>{title}</Typography>*/}
      
      {/*Apply sx prop for custom styling */}
      <Typography sx={{ ...sx, variant: sx?.variant ? sx.variant : "body1" }}>{title}</Typography> 
      <Link to={to} />
    </MenuItem>
  );
};

const Sidebar = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [selected, setSelected] = useState("Dashboard");

  return (
    <Box
      sx={{
        "& .pro-sidebar-inner": {
          background: `${colors.black[400]} !important`, //sets the background color of the sidebar
        },
        "& .pro-icon-wrapper": {
          backgroundColor: "transparent !important", //making the background of the icon wrapper transparent
        },
        "& .pro-inner-item": {
          padding: "5px 35px 5px 20px !important", //sets the padding of the inner items
        },
        "& .pro-inner-item:hover": {
          color: "#6b74fa !important", //changes the text color to a light purple shade when the item is hovered over, making it a visual cue for interactivity.
        },
        "& .pro-menu-item.active": {
          color: "#525dfa !important", //changes the text color of the active menu item to a different shade of blue, typically to signify which page or section is currently active or selected 
        },
      }}
    >
      <ProSidebar collapsed={isCollapsed}>
        <Menu iconShape="square">
          {/* LOGO AND MENU ICON */}
          <MenuItem
            onClick={() => setIsCollapsed(!isCollapsed)}
            icon={isCollapsed ? <MenuOutlinedIcon /> : undefined}
            style={{
              margin: "10px 0 20px 0",
              color: colors.grey[100],
            }}
          >
            {!isCollapsed && (
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                ml="15px"
              >
                <Typography variant={isCollapsed ? "h5" : "h3"} color={colors.grey[100]}>
                  GXS Bank
                </Typography>
                <IconButton onClick={() => setIsCollapsed(!isCollapsed)}>
                  <MenuOutlinedIcon />
                </IconButton>
              </Box>
            )}
          </MenuItem>

          {/* add in log */}
          {!isCollapsed && (
            <Box mb="25px"> 
              <Box display="flex" justifyContent="center" alignItems="center">
                <img //image tag, 
                  alt="logo"
                  width="200px"
                  height="100px"
                  src={"https://static.mycareersfuture.gov.sg/images/company/logos/c91e3fa8d760d9b976e75e078acefeb1/gxs-bank.png"} 
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
              </Box>
            </Box>
          )}

          <Box paddingLeft={isCollapsed ? undefined : "10%"}>
            
            <Item
              title="Dashboard"
              to="/"
              icon={<HomeOutlinedIcon />}
              selected={selected}
              setSelected={setSelected}
              sx={{ fontSize: '1.5rem' }}
            />

            <Typography
              variant="h4"
              color={colors.grey[300]}
              sx={{ m: "15px 0 5px 20px" }}
            >
              Data
            </Typography> 

            <Item
              title="Retention Stategies"
              to="/retention-strategies"
              icon={<OnlinePredictionIcon />}
              selected={selected}
              setSelected={setSelected}
              sx = {{ fontSize: '1rem' }}
            />

            <Typography
              variant="h6"
              color={colors.grey[300]}
              sx={{ m: "15px 0 5px 20px" }}
            />
            
            <Item
              title="Churn Predictions"
              to="/churn-prediction"
              icon={<OnlinePredictionIcon />}
              selected={selected}
              setSelected={setSelected}
              sx = {{ fontSize: '1rem' }}
            />

            <Typography
              variant="h6"
              color={colors.grey[300]}
              sx={{ m: "15px 0 5px 20px" }}
            />

          </Box>
        </Menu>
      </ProSidebar>
    </Box>
  );
};

export default Sidebar;