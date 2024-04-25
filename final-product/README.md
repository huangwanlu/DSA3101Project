# GXS Customer Churn Analysis Web Application

_This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app)._

The design principle behind this dashboard focuses on providing high-level management executives with intuitive, data-driven insights through a clean, executive-friendly interface. This approach prioritizes clear visualizations, easy navigation, and actionable information, ensuring that decision-makers can quickly grasp key metrics and make informed choices without getting bogged down in technical details.

## Overview of '/final-product' folder

In the project directory, you can find the following files:
1. `src/components/`: This folder contains code for graphs and charts on the dashboard.
2. `src/scenes/global/`: This folder contains `Sidebar.jsx` and `Topbar.jsx` which generates the topbar and sidebar.
3. `src/scenes/churn-prediction`: This folder contains `Churn.jsx` and `Churn.css` which generates the Churn Prediction page.
4. `src/scenes/retention-strategies`: This folder contains `Retention.jsx` which generates the Retention Strategies page.
5. `src/App.js`: The code sets up a React application with a flexible theme, color mode context, and routes to different components, providing a structure for a dashboard, churn predictions, and retention strategies.
6. `src/Dashboard.jsx`: This code defines a React component, "Dashboard", that organizes and displays various data visualizations and information related to customer churn, including pie charts, bar charts, and a list of ranked churn drivers, within a interactive grid layout and with features for exporting data.
7. `src/DownloadButton.jsx`: This code defines a `DownloadButton` React component that displays a styled button with a download icon and triggers an export function (`handleExportClick`) when clicked, allowing users to download a report.
8. `src/ExportReport.js`: This code defines a `DataExport` React component that allows exporting the content of a referenced dashboard (`dashboardRef`) to a PDF file using `html2canvas` and `jsPDF`, with a `DownloadButton` to initiate the export.
9. `src/index.css`, `src/index.js`: This code sets up global styles with custom scrollbars and fonts in `index.css`, and initialize a React application with routing functionality in `index.js`, rendering the main App component within a `BrowserRouter`.
10. `src/theme.js`: This code sets up a React context for switching between light and dark themes, defines color tokens for different modes, and provides a custom Material UI theme with configurable typography, allowing users to toggle between light and dark modes with consistent styling across the application.
11. `package-lock.json`: This file captures the exact versions of each dependency, including any sub-dependencies, to create a reproducible build environment. When someone else installs the project (e.g., using `npm install`), npm reads this file to determine which versions of dependencies to install, ensuring consistency.
12. `package.json`: This file locks the exact versions of the dependencies and their transitive dependencies (dependencies of dependencies) at the time they were installed. It ensures that every installation is consistent, reducing the risk of "it works on my machine" issues. It is created and updated by npm when you install or update packages in your project.
13. `react_Dockerfile`: This Dockerfile sets up a Docker image for a React application, installs dependencies, copies the necessary files, exposes port 3000, and sets the command to start the React development server.


For any troubleshooting, refer to Technical Report. 
