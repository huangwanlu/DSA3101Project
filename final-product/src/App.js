import { CssBaseline, ThemeProvider } from "@mui/material";
import { ColorModeContext, useMode } from "./theme";
import Topbar from "./scenes/global/Topbar";
import Sidebar from "./scenes/global/Sidebar";
import { Routes, Route } from "react-router-dom";
import Dashboard from './Dashboard.jsx';
import Churn from "./scenes/churn-prediction/Churn";
import Retention from "./scenes/retention-strategies/Retention.jsx";

function App() {
  const [theme, colorMode] = useMode();

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div className="app">
          <Sidebar />
          <main className="content">
            <Topbar />
            <Routes>
              <Route path = "/" element = {<Dashboard />} />
              <Route path="/churn-prediction" element={<Churn />} />
              <Route path="/retention-strategies" element={<Retention />} />
             </Routes>
          </main>
        </div>
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

export default App;