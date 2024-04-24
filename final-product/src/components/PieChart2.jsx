import React from 'react';
import { ResponsivePie } from "@nivo/pie";
import { useTheme } from "@mui/material";

const PieChart2 = () => {
  const theme = useTheme();

  // Define colors for Male and Female
  const activeColor = '#32b381'; // Use your desired blue color
  const nonactiveColor = "#FF6666"; // Use pink color

  // Define data for Male and Female
  const data = [
    { id: "Active", label: "Active", value: 36.08 }, // Example values, replace with actual data
    { id: "Non-Active", label: "Non-Active", value: 63.92  }, // Example values, replace with actual data
  ];

  return (
    <ResponsivePie
      data={data}
      colors={[activeColor, nonactiveColor]} // Use defined colors for Male and Female
      width={160} 
      height={160} 
      margin={{ top: 20, right: 40, bottom: 60, left: 40 }}
      innerRadius={0.3}
      padAngle={0.7}
      cornerRadius={3}
      activeOuterRadiusOffset={8}
      borderColor={{
        from: "color",
        modifiers: [["darker", 0.2]],
      }}
      enableRadialLabels={false} // Disable radial labels
      enableSliceLabels={true} // Enable slice labels
      enableArcLinkLabels={false} // Disable lines extending from pie chart
      sliceLabel={(datum) => `${datum.value}%`} // Corrected template literal syntax // Display percentage value on the slice
      sliceLabelsSkipAngle={10} // Adjust as needed
      sliceLabelsTextColor="#ffffff" // Set text color for slice labels
      sliceLabelsTextStyle={{ fontSize: '18px', fontWeight: '1000' }} // Customize font size and weight
      tooltip={({ datum }) => (
        <strong>{datum.label}: {datum.value}%</strong>
      )}
      legends={[
  {
    anchor: 'right',
    direction: 'column',
    translateX: 140, // Increase this value to move the legend further to the right
    translateY: 0,
    itemsSpacing: 2,
    itemWidth: 140,  // Adjust if necessary to accommodate longer labels
    itemHeight: 20,
    itemTextColor: '#999',
    itemDirection: 'left-to-right',
    itemOpacity: 1,
    symbolSize: 18,
    symbolShape: 'circle',
    effects: [
      {
        on: 'hover',
        style: {
          itemTextColor: '#000',
        },
      },
    ],
    data: [
      {
        id: 'Active', // Use the ID from the data
        label: 'Y', // Text to display
        color: activeColor, // Color of the symbol
      },
      {
        id: 'Non-Active', // Use the ID from the data
        label: 'N', // Text to display
        color: nonactiveColor, // Color of the symbol
      }
    ],
  },
]}
 // Remove legends since we only have two categories
    />
  );
};

export default PieChart2;