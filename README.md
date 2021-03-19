# Car Price Prediction

An automobile company outisde US aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
Which variables are significant in predicting the price of a car How well those variables describe the price of a car Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market.
Business Goal - You are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

## Data

The data for the following problem is [available on Kaggle.](https://www.kaggle.com/hellbuoy/car-price-prediction/) 
Since the data has been added to the `data/` directory, cloning this repository would suffice.

## Pre-requisites

The project was developed using python 3.8.3 with the following packages.
- Pandas
- Numpy
- matplotlib
- seaborn
- Scikit-learn
- statsmodels


## Files
- notebook/Car_Price_IMP_Feature_Prediction_LinearRegression_Ridge_and_Lasso.ipynb : Jupyter Notebook with all the workings including pre-processing, modelling using Multiple Linear Regression, further applying Ridge and Lasso Regularization and finally inference.

- notebook/Car_Price_Prediction_XGBoost_Regression.ipynb : Jupyter Notebook with all the workings including pre-processing, modelling using XGBoost Regression to achieve better R2 score and inference.

- data/ : The source data and data dictionary.


## Acknowledgements

[Kaggle](https://kaggle.com/), for providing the data for this problem statement.