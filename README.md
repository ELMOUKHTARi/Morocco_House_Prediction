# Morocco House Price Predictor
This project is a house price prediction application that estimates property prices in Morocco based on key features such as total square footage, location, number of bedrooms (BHK), and number of bathrooms. The project uses a dataset specific to Morocco's real estate market to train a machine learning model that predicts house prices for different cities and regions across the country.

## Features
- **Predict House Prices:** Enter features like square footage, location, BHK, and bathrooms to get an estimated house price.
- **Location-Specific Predictions:** Adjusts predictions based on specific cities or regions in Morocco, thanks to one-hot encoding of location data.
- **User-Friendly Web Interface:** Built with the Flask framework, the application provides a simple web interface for users to input data and view predictions.
## Technologies Used
- **Machine Learning:** The model was trained with key features from the Moroccan housing dataset.
- **Flask:** Used to create the web API and interface for the prediction system.
- **JSON & Pickle:** For saving and loading model artifacts and data columns.
## How to Run
### 1- run server :
..\..\saab_project> cd server
..\saab_project\server> python server.py

### 2- go to "..\project\poo\saab_project\client"
Enter the filr "app.html"

Access the web interface to enter property details and get an estimated price.
done...

## if you want to change the dataset to train the model or create your model:

1. run jupyter notebook server

2. open the file "saab_predictor.ipynb"

3. follow step of cleaning your dataset, i use the dataset named "DataPOO.csv", you can change it ase your

4. after finishing cleaning your dataset, you can plot it, so you can visulate it.

5. after that you can spilte your dataset for training and test

6. you can after that choose which model is the best for you and use it 

7. ceate your model with pickle

8. and finally you have to exporting the dataset columns to a json file 

## Web interface by flask framework

![Capture d'écran 2024-11-12 001840](https://github.com/user-attachments/assets/37367fd1-8955-43ed-b645-9d8e82469818)


This project can be helpful for anyone interested in real estate market predictions, especially in Moroccan cities. It demonstrates the integration of machine learning with web development to build practical, user-centered applications.
