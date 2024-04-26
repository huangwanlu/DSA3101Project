# Byteme's Churn Prediction Project

## Introduction

Our project is centred around helping GXS with its customer retention, by successfully predicting whether their customers will churn and 
providing suggestions to reduce churn rate. We did this by implementing machine learning models and evaluating their performance through
evaluation metrics like accuracy.

With all the results generated from our model, we displayed it on our web application, which has been designed to be user-friendly and 
interactive. This allows users to interpret complex data more easily and enables them to seamlessly navigate through our web application.

Users are able to upload their own CSV files, following the instructions that are available in our "Churn Predictions" tab, to generate
the churn prediction results based on their dataset.

## How to Run Our Project

1) Navigate to the directory where our `docker-compose.yaml` file is located on your computer.
2) Ensure Docker Desktop is up and running
3) In your terminal, run this command:
   ```bash
   docker-compose up
   ```
   Note: It takes a while for Docker to create the container. Waiting time depends on individual's computer.
4) Visit `localhost:8000` on your browser to verify that our backend server is running. If no error pops up, it means it is running successfully.
5) Visit `localhost:3000` on your browser to verify that our frontend server is running. If it is running, you should be seeing a web
application with GXS' logo in the top left.

   Note: It takes a while for both `localhost:3000` and `localhost:8000` to start up on web browser. Waiting time depends on individual's computer.
7) To stop running the container, users can either stop directly on Docker desktop or in terminal by pressing `Ctrl + C` on keyboard.
8) To delete the containers and network created, run this command:
   ```bash
   docker-compose down
   ```

## How to Upload Your Own Dataset and Generate Churn Predictions
1) Go to the "Churn Predictions" tab in our frontend server.
2) Hover over the "?" icon to view instructions of what to include in your csv file.
3) Click on "Choose file" and select the file you wish to upload.
4) Click on "Upload CSV File".
5) If file is uploaded successfully, a table with the "Churn" column will be generated in real-time.

