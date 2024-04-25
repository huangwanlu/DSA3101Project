import React from 'react';
import { Label, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from 'recharts';

const data = [
  { Age_Group: '18-29', Churn_Rate: 7.56 },
  { Age_Group: '30-39', Churn_Rate: 10.9 },
  { Age_Group: '40-49', Churn_Rate: 30.8 },
  { Age_Group: '50-59', Churn_Rate: 56 },
  { Age_Group: '60-69', Churn_Rate: 35.2 },
  { Age_Group: '70-79', Churn_Rate: 10.3 },
  { Age_Group: '80-89', Churn_Rate: 7.69 },
  { Age_Group: '90-100', Churn_Rate: 0 }
];

const renderCustomTitle = () => {
  return (
    <text x="50%" y="20" textAnchor="middle" dominantBaseline="central" fontSize="18" fontWeight="bold" fill="#8884d8">
      Churn Rate by Age Group
    </text>
  );
};

const AgeChurnBarChart = () => (
  <ResponsiveContainer width="100%" height={280}>
    <BarChart
      data={data}
      margin={{
        top: 40, right: 30, left: 20, bottom: 15,
      }}
    >
      {renderCustomTitle()}

      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="Age_Group" interval={0}>
        <Label value="Age Group" offset={-10} position="insideBottom" />
      </XAxis>
      <YAxis label={{ value: 'Churn Rate (%)', angle: -90, position: 'insideLeft' }} />
      <Tooltip />

      <Bar dataKey="Churn_Rate" fill="#8884d8">
  <LabelList dataKey="Churn_Rate" position="insideBottom" fill="#000" />
</Bar>
    </BarChart>
  </ResponsiveContainer>
);

export default AgeChurnBarChart;