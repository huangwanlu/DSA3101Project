import React from 'react';
import { Typography, List, ListItem, Box } from '@mui/material';

const Retention = () => {
  return (
    <Box m={4}>
      {/* Page Title */}
      <Typography
        variant="h2" // Adjust the variant for desired size
        style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }} // Adjust fontSize and marginBottom as needed
      >
        Customer Churn Retention Strategies
      </Typography>
      
      <Typography
        variant="h4"
        style={{ fontSize: '15px', fontWeight: 'bold' }}
        mt="0px"
        mb={4}
      >
        Feature importance scores tell you which features were most useful for predicting the outcome. Given that we have implemented the Random Forest Algorithm via Scikit-Learn, these scores represent the average improvement in prediction clarity brought on by splitting data based on each feature across all the trees in the forest. Features that split the data more effectively have higher scores, indicating greater importance for accurate predictions.
      </Typography>
      {/* GRID & CHARTS */}
      <Box
        display="grid"
        gridTemplateColumns="repeat(12, 1fr)"
        gridAutoRows="140px"
        gap="20px"
      >
        {/* ROW 1 */}
        <Box
          gridColumn= "span 12"
          backgroundColor="#6E7A9D"
          display="flex"
          alignItems="center"
          justifyContent="flex-start"
        >
          {/* Nested List */}
          <List
            component="nav"
            aria-label="Feature scores"
            style={{ lineHeight: '1.2', margin: 0, padding: '24px' }}
          >
            <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.3rem' }}>1. Age: 0.236</ListItem>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • For the young customers, utilize mobile apps, social media, and gamification to keep the brand engaging. </Typography>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • For middle-aged customers, develop reward programs that offer real value based on customer purchase history and preferences.</Typography>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • For older customers, offering a support hotline and service representatives ensures clear communication and easy accessibility, as they value high-quality customer service and personal interaction. Additionally, banks can host workshops to help them master digital banking tools, thus bridging the technology gap.</Typography>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Offer financial products tailored to the primary financial concerns of elders, such as retirement planning, estate management, and healthcare financing.</Typography>
          </List>
        </Box>
        {/* ROW 2 */}
        <Box
          gridColumn= "span 12"
          backgroundColor="#6E7A9D"
          display="flex"
          alignItems="center"
          justifyContent="flex-start"
        >
          {/* List items */}
          <List
            component="nav"
            aria-label="Feature scores"
            style={{ lineHeight: '1.2', margin: 0, padding: '24px' }}
          >
            <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>2. Credit Score: 0.147</ListItem>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Introduce reward programs that incentivize improving or maintaining a high credit score. For example, offer reduced interest rates or higher credit limits to customers who achieve score improvements or maintain excellent credit standings.</Typography>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Offer tailored credit counseling services to help customers understand and improve their credit scores, such as providing them with detailed monthly reports and recommendations on how to manage their debts, bills, and credit lines more effectively.</Typography>
          </List>
        </Box>
        {/* ROW 3 */}
        <Box
          gridColumn= "span 12"
          backgroundColor="#6E7A9D"
          display="flex"
          alignItems="center"
          justifyContent="flex-start"
        >
          {/* List items */}
          <List
            component="nav"
            aria-label="Feature scores"
            style={{ lineHeight: '1.2', margin: 0, padding: '24px' }}
          >
            <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>3. Estimated Salary: 0.146</ListItem>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Provide financial advisory services that help customers plan according to their salary levels. For higher earners, advice can be oriented towards investment and saving for wealth growth. For mid to lower earners, the focus can be on budget management, debt reduction, and building emergency savings.</Typography>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Offer banking products that are customized to the salary bands of customers. For instance, higher salary customers might be offered premium accounts with exclusive benefits, while lower salary customers might receive offers for high-interest savings accounts that help their money grow.</Typography>
          </List>
        </Box>
        {/* ROW 4 */}
        <Box
          gridColumn= "span 12"
          backgroundColor="#6E7A9D"
          display="flex"
          alignItems="center"
          justifyContent="flex-start"
        >
          {/* List items */}
          <List
            component="nav"
            aria-label="Feature scores"
            style={{ lineHeight: '1.2', margin: 0, padding: '24px' }}
          >
            <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>4. Balance: 0.138</ListItem>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Reward customers who maintain or grow their balances within a certain period with loyalty bonuses such as cash back, additional interest, or one-time credit to their account. </Typography>
            <Typography variant="body2" style={{ fontSize: '14px' }}> • Offer proactive tips when customers' balance drops to a certain threshold and reward notifications when they reach saving milestones.</Typography>
          </List>
        </Box>
        {/* ROW 5 */}
        <Box
          gridColumn= "span 12"
          backgroundColor="#6E7A9D"
          display="flex"
          alignItems="center"
          justifyContent="flex-start"
        >
          {/* List items */}
          <List
            component="nav"
            aria-label="Feature scores"
            style={{ lineHeight: '1.2', margin: 0, padding: '24px' }}
          >
              <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>5. Products Number: 0.134</ListItem>
        <Typography variant="body2" style={{ fontSize: '14px' }}> • Introduce a "Grow Your Financial Suite" rewards program that offers tiered incentives for each new product adopted. For example, customers could receive a one-time cash bonus for opening a new type of account or enrolling in a new service. </Typography>
        <Typography variant="body2" style={{ fontSize: '14px' }}> • Provide additional perks for continuous engagement with new products, such as interest rate boosts on savings accounts, fee waivers on additional account services, or exclusive access to financial seminars and webinars for multi-product customers. </Typography>
        </List>
        </Box>
        {/* ROW 6 */}
        <Box
          gridColumn= "span 12"
          backgroundColor="#6E7A9D"
          display="flex"
          alignItems="center"
          justifyContent="flex-start"
        >
          {/* List items */}
          <List
            component="nav"
            aria-label="Feature scores"
            style={{ lineHeight: '1.2', margin: 0, padding: '24px' }}
          >
            <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>6. Tenure: 0.081</ListItem>
        <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>7. Active Member: 0.044</ListItem>
        <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>8. Country: 0.0379</ListItem>
        <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>9. Credit Card: 0.019</ListItem>
        <ListItem style={{ fontWeight: "1000", color: "#2E1A47", paddingBottom: '2px', paddingTop: '2px', fontSize: '1.2rem' }}>10. Gender: 0.0176</ListItem>
          </List>
          </Box>
      </Box>
    </Box>
  );
};

export default Retention;