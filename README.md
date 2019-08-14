## Sales-Per-Day-Prediction

# Data Preprocessing and Feature Selection
Raw dataset consist of total seven independent features with target variable (dependent). Before dive into to the machine learning (ML) model it's necessary to prepare raw data and make it switable for the machine learning (ML) algorithm.

Data cleaning process needs to be done at first beacuse raw dataset may have redundant data which is not a good idea to train with ML model. For instance, in the raw dataset customers_per_day and sales_per_day columns never be nagative that's why it's no a good idea to keep it into the processed dataset also dropping out the NaN (not a number) values which has been replaced with the mean or zero values and so on.

Then features selection has done for choosing the best features among all the processed data features. Some advantages for doing feature selection is, identify most and least relevant features, prevent from data overfitting, minimize computational time, cost and maximize prediction accuracy etc. Mutual information basically calculate the mutual dependence between two random variables that is how similar the joint distribution of the pair.

Raw data consist of seven independent features brand, country, currency, customers_per_day, outlet_id, week_id and weekday. sales_per_day is the target variable. Procesed data features consist of numeric and text data values, but for ML algorithm it's necessary to encoded the text data to numerical data. Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form.

# Mutual Information (MI)
This part it's necessary to see which data features are the most important features, MI calculate how similar the joint distribution of the pair. Each time mutual information takes one input data feature and finding the depedence with target variable. It's gives a score (0 to 1) for each input data features. Zero score means two random variable are independent and one means they are completely dependent with each other.

# Machine Learning Model
For solving regression problem many machine learning algrithm has been repidly used like support vector mahchine, ada boost, decision tree, neural net etc. But in this project complete regression model is presented with neural netork (NN). Precisely, a simple neural net with 2 hidden layer with 130 neurons for each layer. In term of activation fucction relu has been used for non-linearity in between layers. Also each epoch during training time 30% of nodes randomly switched off (dropout), this technique basically force all the hidden nodes to contribute during the training time and eventually it's also effective to prevent data overfitting as well.

# Evaluation & Outlook
Finally for testing part, test data which has seperated before model training has been used for calculating the model accuracy, that means how well model has learned from the previous data pattern. Two common matrices used in this part root mean square error (rmse) and mean abosolute error (mae). Test dataset with neural net trained model showed over 97% accuracy based on mean absolute error.

