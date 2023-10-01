# Lux_Week_1_project

### Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that. So, if you were in charge of predicting customer churn how would you go about using machine learning to make a good guess about which customers might leave? Like, what steps would you take to create a machine learning model that can predict if someone's going to leave or not?

> * In determining the success rate of Instagram TV products, some of the metrics I would use are:
>   
> * Given the data about the customers who left and the items they bought, along with their tendencies. This would be retrieved from the data set given after cleaning and feature engineering. I would use these factors as key labels in the analysis of the data that would be provided.

        # Import the libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load the data
        df = pd.read_csv('customer_churn.csv')
        
        # Clean the data
        # Here you can perform data cleaning operations, such as handling missing values, removing duplicates, and data type conversions.
        # Remove rows with missing values
        df.dropna(inplace=True)
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Convert a column to a different data type, e.g., 'TotalCharges' to float
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        
        # Engineer features
        # Feature engineering involves creating new features from existing ones or transforming existing features.
        # Calculate the ratio of 'TotalCharges' to 'MonthlyCharges'
        df['ChargesRatio'] = df['TotalCharges'] / df['MonthlyCharges']
        
        # Create a binary feature indicating whether a customer has a long-term contract
        df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)
        
        # You can add more feature engineering steps based on your specific dataset and objectives.
        
        # Save the cleaned and engineered DataFrame to a new CSV file if needed
        df.to_csv('cleaned_and_engineered_data.csv', index=False)


>  * The next step would be to split this data to obtain a set I can train on and a set that I would test on called a held-out set.
     from sklearn.model_selection import train_test_split

        # Define the features (X) and the target variable (y)
        # Assuming 'Churn' is your target variable
        X = df.drop(columns=['Churn'])  # Features
        y = df['Churn']  # Target variable
        
        # Split the data into training and test sets (e.g., 80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 'X_train' and 'y_train' will be your training data.
        # 'X_test' and 'y_test' will be your held-out (test) data.
        
        # You can adjust the 'test_size' parameter to specify the proportion of data to use for testing.
        # 'random_state' ensures reproducibility of the split, and you can set it to any integer value.
        
        # Now you can use X_train and y_train to train your machine learning model
        # and X_test and y_test to evaluate its performance.
        
        
             * I would also look at what features are most important. 
        
             from sklearn.ensemble import RandomForestClassifier
        
        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier(random_state=42)
        
        # Fit the model on the training data
        rf_classifier.fit(X_train, y_train)
        
        # Get feature importances
        feature_importances = rf_classifier.feature_importances_
        
        # Associate feature importances with column names
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
        
        # Sort features by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        # Print or visualize feature importances
        print(feature_importance_df)

     
> *Thereafter I would use these features to create a machine learning model that would take these features against a key feature of churn or not churn.

             from sklearn.linear_model import LogisticRegression
        
        # Create a Logistic Regression model
        logistic_model = LogisticRegression(random_state=42)
        
        # Fit the model on the training data using the selected features
        important_features = ['Feature1', 'Feature2', 'Feature3']  # Replace with your important features
        logistic_model.fit(X_train[important_features], y_train)
        
        # Predict churn on the test data
        y_pred = logistic_model.predict(X_test[important_features])
        
        # Evaluate the model's performance
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        print(f'Accuracy: {accuracy}')
        print('Confusion Matrix:\n', confusion)
        print('Classification Report:\n', classification_rep)

 
> * My model would be a classification model that would be aimed at doing a diagnostic-level analysis. I would use a base model, such as logistic regression, and continue to finetune for accuracy with better-performing models such as Random Forest or XG-Boost.
        
             from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        
        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier(random_state=42)
        
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train[important_features], y_train)
        
        # Get the best parameters and best estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        
        # Fit the model with the best hyperparameters on the training data
        best_estimator.fit(X_train[important_features], y_train)
        
        # Predict churn on the test data
        y_pred_rf = best_estimator.predict(X_test[important_features])
        
        # Evaluate the Random Forest model
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        confusion_rf = confusion_matrix(y_test, y_pred_rf)
        classification_rep_rf = classification_report(y_test, y_pred_rf)
        
        print(f'Random Forest Accuracy: {accuracy_rf}')
        print('Random Forest Confusion Matrix:\n', confusion_rf)
        print('Random Forest Classification Report:\n', classification_rep_rf)
        
        
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV
        
        # Create an XGBoost classifier
        xgb_classifier = xgb.XGBClassifier(random_state=42)
        
        # Define hyperparameters and their possible values for tuning
        param_grid_xgb = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.001]
        }
        
        # Perform grid search with cross-validation
        grid_search_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb, cv=5, scoring='accuracy')
        grid_search_xgb.fit(X_train[important_features], y_train)
        
        # Get the best parameters and best estimator
        best_params_xgb = grid_search_xgb.best_params_
        best_estimator_xgb = grid_search_xgb.best_estimator_
        
        # Fit the XGBoost model with the best hyperparameters on the training data
        best_estimator_xgb.fit(X_train[important_features], y_train)
        
        # Predict churn on the test data
        y_pred_xgb = best_estimator_xgb.predict(X_test[important_features])
        
        # Evaluate the XGBoost model
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        confusion_xgb = confusion_matrix(y_test, y_pred_xgb)
        classification_rep_xgb = classification_report(y_test, y_pred_xgb)
        
        print(f'XGBoost Accuracy: {accuracy_xgb}')
        print('XGBoost Confusion Matrix:\n', confusion_xgb)
        print('XGBoost Classification Report:\n', classification_rep_xgb)


> * After finding the best-performing model, preferably with an accuracy of above 80%, I would then deploy the model for production so that Sprint would be able to use it for real-time identification of customers likely to churn.

         import pickle
        
        # Serialize and save the trained model
        with open('churn_prediction_model.pkl', 'wb') as model_file:
            pickle.dump(best_estimator, model_file)
        from flask import Flask, request, jsonify
        import pickle
        
        app = Flask(__name__)
        
        # Load the serialized model
        with open('churn_prediction_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        @app.route('/predict', methods=['POST'])
        def predict_churn():
            try:
                data = request.get_json()
                features = data['features']  # Provide the input data in the request body
        
                # Make predictions using the loaded model
                predictions = model.predict([features])
                prediction_probabilities = model.predict_proba([features])
        
                response = {
                    'prediction': predictions[0],
                    'probability': prediction_probabilities[0][1]  # Probability of churn
                }
        
                return jsonify(response)
        
            except Exception as e:
                return jsonify({'error': str(e)})
        
        if __name__ == '__main__':
            app.run(host='0.0.0.0', port=5000)
