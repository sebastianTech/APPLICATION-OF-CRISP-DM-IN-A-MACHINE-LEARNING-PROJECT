# 
APPLICATION-OF-CRISP-DM-IN-A-MACHINE-LEARNING-PROJECT
“ Shawn broker is a company that deals in home loans in London. They have their presence in 32 London borough councils in London. When customers approach them for a loan, they validate the credibility of the customers which will determine their eligibility for a loan. The volume of loan they process is on geometric increase, hence automating the process involved in accessing the loan eligibility in real-time is a necessity in with the information customers have already provided when they filled the application form. The entry in the form includes gender, marriage status, education qualification or level, number of person living with them, annual income, loan amount being requested, credit worthy of the applicant. Automating the process is possible, however, they have a challenge that needs to be tackled first, which is to understand the customer features qualified for loan and the amount which will help them to specify their target customers. This forms the basis of my research question”.
Phase 2
Data Understanding/ background information about the chosen data set 
We import python packages and libraries such as pandas, seaborn and sKlearn. This project contains 3 files (Train, Test, and sample_submission). The train file is where the training will be done, test file is where the model will be applied to predict the target variable and the sample_submission is where the predicted outcomes will be submitted. The features or variables present in the train and test data sets are 
Train and Test:
[‘Loan_ID’, ‘Gender’, ‘Married’, ‘Dependents’, ‘Education’, ‘Self_Employed’, ‘Application_Income’ ‘CoapplicantIncome’, ‘LoanAmount’, ‘Loan_Amount_Term’, ‘Credit_History’, ‘Property_Area’, ‘Loan_Status’)
The Test data contains all the variables except “Loan Status” which is what will be predicting through the model that will be built on train data.
![image](https://user-images.githubusercontent.com/66043834/162639769-437a4629-09af-4981-b4ad-0a853e094a06.png)



This a classification problem geared towards predicting if a loan will be approved or not. It requires predicting a discrete value based on a given data set of independent variables.
Hypothesis Generation
The next phase in understanding the business problem is to generate some hypothesis, by looking out the factors that can affect the impact of the income. From the variable, I think the below factors can affect the variables
	Salary: Obviously, customers with high salaries are likely to get approval for their loan requests.
	Previous History: Likewise, customers who have the antecedents of repaying their loan stand a chance of their loan request being approved.
	Loan Amount: smaller loan should be approved but again, it depends on the length of repayment.
	Loan term: A Loan with less time to repay should be considered first before the rest.
All of the above factors will be considered in this task.
Data Exploration
![image](https://user-images.githubusercontent.com/66043834/162639795-d5e75e2e-ec12-4ab9-8f00-e462ea84f763.png)

There are 12 independent variables and 1 target variable in the train and Test data.
 ![image](https://user-images.githubusercontent.com/66043834/162639810-1b941440-ed65-494f-93f2-fea18f80b519.png)

Fig 2: Train and Test Data Exploration.
 
 ![image](https://user-images.githubusercontent.com/66043834/162639820-59ab0012-9c58-4b09-89ec-8d2629165fa8.png)

Fig 3: Data type/ shape in Train dataset
Exploring the target variable showed that 422 people out of 614 got their loan approved which made a total of 69% as shown below.
 
 ![image](https://user-images.githubusercontent.com/66043834/162639844-468e9ba3-745d-4b79-a957-495f413c755c.png)

Fig 4: Visualisation of Target variable (‘Loan_Status’)
Univariate Analysis
Analysis of variables based on their data type: Categorical, Ordinal and numerical). For categorical data visualising is done with bar plots while the numerical features and its probability density plots will be used to visualise its distribution.
Independent Variable (Categorical)
In fig 5 below:
	65% are married applicant,
	 15% of the applicant are employed and,
	 85% have a good credit history.

![image](https://user-images.githubusercontent.com/66043834/162639849-d90c9c44-dffc-4f31-be7e-3bb8532d4814.png)


Fig 5: Independent Variable Visualisation (Categorical data type).
Independent Variable (Ordinal)
The inference from fig 6 below shows that majority of the loan applicant don’t have dependents, 80% are graduate and a hand full of the applicant are from the semi-urban area. 
 ![image](https://user-images.githubusercontent.com/66043834/162639860-13af5a83-dcce-4d43-b119-6789f13b0782.png)

Fig 6:  Independent Variable Visualisation (Ordinal data type).

Independent Variable (Numerical)
Having seen the categorical and ordinal variables, fig 7 below shows that the distribution between the Applicant incomes is not evenly distributed because algorithms work better when the data points are evenly distributed. This shows that there are outlier and majority of the graduate with high incomes appears to be part of the outliers. 
  ![image](https://user-images.githubusercontent.com/66043834/162639869-0f8d60f6-95b8-45e3-ae9f-00d865318fba.png)
  
Fig 7: Independent Variable Visualisation (Numerical data type).
Bivariate Analysis
The hypothesis generated from the data will be tested in this session. With bivariate analysis, we will see the relationship between the variables and the target variable.
 Independent Variables (Categorical data type) vs Target Variable
For approved and unapproved loans, the ratio of male and female who applied are the same, a good number of married loan applicants were approved, and no inference can be drawn from self_employed. 
 ![image](https://user-images.githubusercontent.com/66043834/162639876-ac3f505b-f808-4196-a82f-c3b5567fe4be.png)
    
Fig 8: Categorical Independent Variables vs Target Variable

 Independent Variable (Numerical data type) vs Target Variable
In fig 9 below, Applicant income is not significant to the chances of getting a loan approved which knocks off one of the hypothesis meaning that someone earning high can get the loan application disapproved. The loan amount reveals that small loan have a high chance of being approved and that validates one of the hypothesis. The less the loan amount, more chance of being approved.
 ![image](https://user-images.githubusercontent.com/66043834/162639891-e5b79f68-fdc9-4baa-ad3b-74afbda4c70d.png)
    
Fig 9: Numerical independent variable vs Target variable
The correlation analysis below shows a good correlation between (‘ApplicantIncome’ & ‘LoanAmount’), (‘Credit_History’ & ‘Loan_Status’) and (‘LoanAmount’ & ‘CoapplicantIncome’)
![image](https://user-images.githubusercontent.com/66043834/162639917-642d16e8-aaf7-4750-b72d-07f41705a7a6.png)

 
Fig: 10 Correlation Analysis

Phase 3
Data Preparation
This is where 80% of the project is done. Based on the insight gotten, the dataset will be cleaned, transformed, integrated and formatted for model build.
Treatment of Missing Values
From the fig below, our data contains missing values and if not treated can adversely affect my model performance. The treatment of missing values is as follows:
	For numerical values, imputation is by mean or median.
	Categorical variables will be done using mode.
    ![Uploading image.png…]()
                            
Fig 11a: Missing Values                                            Fig 11b: After treating missing values
Outlier Treatment
This is an important step because the presence of outliers can change the dynamics of the mean, standard deviation, and its distribution spread. The presence of outliers made the dataset to be skewed to the left as seen from Fig 7. To remove this, a log transformation is done, which affects or reduces the large values without affecting the smaller values as shown below. Which looks closer to a normal distribution.


 
Fig 12: Log transformation of the dataset.
Phase 4

Model Build (Logistic Regression)
Regression is a predictive technique that shows the relationship between a dependent (target) and an independent variable (predictor). There are various regression techniques, the right choice is determined by three metric:
	number of independent  variables,
	Type of dependent and 
	The Shape of the regression line.
We used logistic because we want to show the probability of discovering customer segments who are likely to get a loan. In other words, the dependent variable is binary (0/1, true or False).
Consider 
Odds = p/(( 1-p))                                                                                                                  (1)                                                                               				         
Where p = probability of customers getting loan;
( 1-p) = probability of them not getting loan.
Since this is a binomial distribution (dependent variable), it is important to pick a link that will best describe the distribution, hence taking the log function of Equ 1 which will minimise the likelihood of observing sample values rather than its sum of squared errors.
Log (odds) = log [p/(( 1-p))  ]                                                                                                 (2)           
                                                                                                                   
Logit (p) = log [p/(( 1-p))  ] = b_0+ b_1 X_1+ b_2 X_2 +b_3 X_3 ……. + b_K X_K                                     (3)                                                                                                                                          
In this project, categorical variables were turned into dummies. The model was trained and tested. All of the above was done from scikit-learn, an open-source library. Prediction on loan status for the validation set gave an accuracy of 0.82.
 
Fig 13: Prediction Accuracy
Phase 5
Evaluation/Validation
This stage is very important because the robust nature of the model will be tested. In this project, we used stratified k-fold cross-validation. Which is the process of re-arranging the data to ensure that each fold is a good representative of the whole dataset. It is a good way of dealing with bias and variance. The purpose of using stratified k-fold cross-validation is to assess the model.
       
Fig 14: Validation accuracy and Roc curve accuracy 

The mean validation is 0.80, the roc curve accuracy of 0.70

Conclusion
In the Jupiter notebook, Decision tree algorithm a supervised algorithm good for classification problems was used, Random forest based on bootstrapping algorithm was used, and grid search was used to optimised the values of the hyperparameters, XGbost was also used. After trying the 4 algorithm, Logistic regression gave the best accuracy followed by Random forest as seen in the notebook.
In this project, the research question which is the understanding of their customer segment, was seen in the bivariate analysis, their eligibility criteria is seen in the correlation analysis and insight to amount of loan that are likely to be approved and the future importance of probability of loan being approved in Appendix A.  Credit history of the customer is the key future to be considered, afterwards balance income, Total income in predicting loan status.  This model gave an accuracy of 80% on the prediction of the likely hood that a customer application for a loan will be approved.
















References

	B. Boser, I. Guyon and V. Vapnik, “A training algorithm for optimal margin classifiers” COLT 1992: Proceedings of the Fifth AnnualWorkshop on Computational Learning Theory, pp. 144-152, 1992.
	I. H. Witten and E. Frank, “Data Mining: Practical Machine Learning Tools and Techniques” Morgan Kaufmann, San Francisco, 2005.
	What is Logistic Regression?,” [Online]. Available: https://www.statisticssolutions.com/what-is-logistic-regression/. [Accessed 06 05 2021].
