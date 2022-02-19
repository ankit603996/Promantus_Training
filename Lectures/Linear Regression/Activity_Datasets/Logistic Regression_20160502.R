#############################
##Null Deviance- is computed using intercept as parameter
##Residual Deviance- is computed with all the parameters in the model
##Loglikelihood is a way of estimating the probabilities given observations
##read GermanCreditData.csv
setwd("F:/Insofe Academics/Cognizant/20160502_Regression/Activity_Datasets")
GermanCreditData<-read.csv("GermanCreditData.csv",header=T,sep=",")
summary(GermanCreditData)
GermanCredit_Numeric <- subset(x = GermanCreditData, select = c("DURATION", "AGE", 
                                                                "AMOUNT"))
GermanCredit_Categorical <- GermanCreditData[,-c(2,14,23)]
str(GermanCredit_Numeric)
# formatize data
GermanCredit_Numeric <- data.frame(apply(X = GermanCredit_Numeric, MARGIN = 2, 
                                         FUN = as.character))
GermanCredit_Numeric <- data.frame(apply(X = GermanCredit_Numeric, MARGIN = 2, 
                                         FUN = as.numeric))
  
# call library
install.packages("infotheo")
library(infotheo)

# bin numeric variables
Discretize_DURATION <- discretize(X = GermanCredit_Numeric$DURATION, 
                                  disc = "equalfreq", nbins = 5)
names(Discretize_DURATION) <- c("Binned_DURATION")
Discretize_AGE <- discretize( GermanCredit_Numeric$AGE, 
                              disc = "equalfreq", nbins = 5)
names(Discretize_AGE) <- c("Binned_AGE")
Discretize_AMOUNT <- discretize(X = GermanCredit_Numeric$AMOUNT, 
                                disc = "equalfreq", nbins = 5)
names(Discretize_AMOUNT) <- c("Binned_AMOUNT")

# merge descritized data with main data
GermanCredit <- data.frame(GermanCredit_Categorical[,-1], 
                           Discretize_DURATION, 
                           Discretize_AGE,
                           Discretize_AMOUNT)

# formating data
GermanCredit <- data.frame(apply(X = GermanCredit, MARGIN = 2, FUN = as.factor))

# remove NA from data
GermanCredit_Final <- na.omit(GermanCredit)

# check for NA's in data
sum(is.na(GermanCredit_Final))

# remove unwanted files from environment
rm(Discretize_AGE, Discretize_AMOUNT, Discretize_DURATION, GermanCredit_Part_2)
rm(GermanCredit, GermanCredit_Categorical, GermanCredit_Numeric, GermanCredit_Part_1)

# split rows into test and train
Rows <- seq(from = 1, to = nrow(GermanCredit_Final), by = 1)
set.seed(1234)
TrainRows <- sample(x = Rows, size = nrow(GermanCredit_Final) * 0.8)
TestRows <- Rows[-TrainRows]

# create train and test data set
TrainData <- GermanCredit_Final[TrainRows,]
TestData <- GermanCredit_Final[TestRows,]
table(TrainData$RESPONSE)
# remove files from environment
rm(Rows, TestRows, TrainRows)


##How do we compute all these
##Logistic regression using only one attribute
logreg_test<-glm(RESPONSE~NEW_CAR,family="binomial",data=TrainData)
summary(logreg_test) ##The coefficients observed are in log odds which are to be converted
predict(logreg_test, type="response")
##Computing probabilities
#prob=exp(coeff)/(1+exp(coeff))
#Therefore for any record we get the probability and we fix a threshold above which is success and below is fail

##Computing deviances
#The residuals observed in logreg_test$resid are of type "deviance"

##NUll Deviance
#The log odds of intercept is 0.91864. we compute corresponding prob
exp(0.91864)/(1+exp(0.91864))
Alpha<-data.frame(response=TrainData$RESPONSE,prob=rep(0.7147649,length(TrainData$RESPONSE)))
Alpha$dev<-ifelse(Alpha$response==1,sqrt(-2*log(Alpha$prob)),-sqrt(-2*log(1-Alpha$prob))) ## To compute deviance

## Observe 
sum(Alpha$dev^2) #Which is approxmately equal to NULL Deviance value given in summary

##Residual Deviance
x<-data.frame(response=TrainData$RESPONSE,prob=logreg_test$fitted.values)
x$err<-ifelse(x$response==1,log(x$prob),log(1-x$prob)) ## 
sum(x$err)
#observe the logLik(logreg_test) and sum(x$err). Are these same
x$Dev<-ifelse(x$response==1,sqrt(-2*log(x$prob)),-sqrt(-2*log(1-x$prob))) ## Gives residual deviance
sum(x$Dev^2) # Observe this value and Residual Deviance given in the summary output model
summary(x$Dev) ## observe this value and Residuals given in the summary output of model

##Applying logistic regression with all variables to predict response
LogReg_Model <- glm(formula = RESPONSE ~., 
                    data = TrainData,family="binomial")

# study summary of model
summary(LogReg_Model)
head(LogReg_Model$fitted.values)

##Applying VIF and StepAIC for important attributes
library(car)
vif(LogReg_Model)

library(MASS)
stepAIC(LogReg_Model)

LogReg_updated<-glm(formula = RESPONSE ~ NEW_CAR + EDUCATION + CO_APPLICANT + 
                      GUARANTOR + PROP_UNKN_NONE + OTHER_INSTALL + RENT + FOREIGN + 
                      CHK_ACCT + HISTORY + SAV_ACCT + PRESENT_RESIDENT + Binned_DURATION + 
                      Binned_AMOUNT, family = "binomial", data = TrainData)


summary(LogReg_updated)

# predict naive model on test data
Predict <- predict(object = LogReg_Model, 
                   newdata = TrainData[,-19],type="response")

Predict[Predict>0.5]=1
Predict[Predict<=0.5]=0

##Test Model efficency using confusion matrix
# build confusion matrix
ConfusionMatrix <- table(TrainData[,19], Predict)
ConfusionMatrix

# calculate Accuracy, Precision & Recall on test data
Accuracy <- sum(diag(ConfusionMatrix))/sum(ConfusionMatrix) * 100
Accuracy

Precision <- ConfusionMatrix[2,2]/(ConfusionMatrix[2,2]+ConfusionMatrix[1,2]) * 100
Precision

Recall <- ConfusionMatrix[2,2]/(ConfusionMatrix[2,2]+ConfusionMatrix[2,1]) * 100
Recall


##Test Model efficency using ROC curves

library(ROCR)
library(ggplot2)
predicted <- predict(LogReg_Model,type="response")
prob <- prediction(predicted, TrainData$RESPONSE)
tprfpr <- performance(prob, "tpr", "fpr")
#A<-performance(prob,"auc")
plot(tprfpr)
str(tprfpr)
cutoffs <- data.frame(cut=tprfpr@alpha.values[[1]], fpr=tprfpr@x.values[[1]], 
                      tpr=tprfpr@y.values[[1]])
tpr <- unlist(slot(tprfpr, "y.values"))
fpr <- unlist(slot(tprfpr, "x.values"))
auc <- performance(prob,"auc")
auc <- unlist(slot(auc, "y.values"))
roc <- data.frame(tpr, fpr)
ggplot(roc) + geom_line(aes(x = fpr, y = tpr)) + 
  geom_abline(intercept=0,slope=1,colour="gray") + 
  ylab("Sensitivity") +    xlab("1 - Specificity")





