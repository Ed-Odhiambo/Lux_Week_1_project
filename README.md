# Lux_Week_1_project

## Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that. So, if you were in charge of predicting customer churn how would you go about using machine learning to make a good guess about which customers might leave? Like, what steps would you take to create a machine learning model that can predict if someone's going to leave or not?

> * In determining the success rate of Instagram TV products, some of the metrics I would use are:
>   
     * Given the data about the customers who left and the items they bought, along with their tendencies. This would be retrieved from the data set given after cleaning and feature engineering. I would use these factors as key labels in the analysis of the data that would be provided.

     * The next step would be to split this data to obtain a set I can train on and a set that I would test on called a held-out set.

     * I would also look at what features are most important. Thereafter I would use these features to create a machine learning model that would take these features against a key feature of churn or not churn.
 
     * My model would be a classification model that would be aimed at doing a diagnostic-level analysis. I would use a base model, such as logistic regression, and continue to finetune for accuracy with better-performing models such as Random Forest or XG-Boost.

     * After finding the best-performing model, preferably with an accuracy of above 80%, I would then deploy the model for production so that Sprint would be able to use it for real-time identification of customers likely to churn.
