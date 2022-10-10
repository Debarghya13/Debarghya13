#important library is caTools
#Calling the library
library(caTools)
#making the directory to store the data by using setwd
setwd("E:/Datasets/College Admission Dataset Logistic Regression")
#reading the data from the csv file using read.csv
my_data = read.csv("College_admission.csv")
my_data
#Splitting the data into two sets one is train set on which we have to train the data and the other on eis test set
#test set is actually hidden from the model and used at the end to predict the accuracy
split = sample.split(my_data, SplitRatio = 0.8)
train = subset(my_data, split == "TRUE")
test = subset(my_data, split == "FALSE")
#munging the data
#here we change admit and rank data from integer to categorical data
#at first these two are integer but after munging these two become factors
my_data$admit <- as.factor(my_data$admit)
my_data$rank <- as.factor(my_data$rank)
#Now we have to fit linear model on our dataset. We are going to use glm function to do so.
#In the below function admit is dependent variable and gpa+rank is independent variable.
#The '~' sign is used to show their relation, we used train set as our data, and family is binomial
#for using logistic regression
my_model<-glm(admit~gpa+rank, data = train, family = 'binomial')
summary(my_model)
#Run the test data through the model
res<-predict(my_model, test, type="response")
res


res<-predict(my_model, train, type="response")
res
#Now we have to validate the model using confusion matrix.
confusion_matrix<-table(Actual_value=train$admit, predict_value = res>0.5)
confusion_matrix
#Confusion matrix gives us the understanding of how our predicted value is alligned with the actual value.
#Understanding the accuracy of our model using Accuracy score
(confusion_matrix[[1,1]] + confusion_matrix[[2,2]])/sum(confusion_matrix)

