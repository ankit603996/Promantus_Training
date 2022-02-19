##################Linear Regression with one variable##############
setwd("F:/Insofe Academics/Cognizant/20160502_Regression/Activity_Datasets")
bigmac <- read.csv("BigMac-NetHourlyWage.csv", header = T, sep = ",")
summary(bigmac)
names(bigmac)[2:3]<-c("Price","Wage")
bigmaclm <- lm(bigmac$Wage~bigmac$Price,bigmac)
summary(bigmaclm)
plot(bigmac$Price,bigmac$Wage,xlab="Price",ylab="Wage")
abline(bigmaclm)  #This gives the regression line on the data

predict <- predict(bigmaclm, bigmac)
predict
yminusyhatsqre <- (bigmac$Price-predict)^2
SSE <- sum(yminusyhatsqre[1:nrow(yminusyhatsqre)])
##Summary interpretation and how we obtain each of the values
Pred<-data.frame(predict(bigmaclm, bigmac, interval="confidence",level=0.95))
Pred_pred<-data.frame(predict(bigmaclm, bigmac, interval="prediction",level=0.95))
par(mfrow=c(1,1))
plot(bigmac$Price,bigmac$Wage) 
points(bigmac$Price,Pred$fit,type="l", col="red", lwd=2) 
points(bigmac$Price,Pred$lwr,pch="-", col="red", lwd=4) 
points(bigmac$Price,Pred$upr,pch="-", col="red", lwd=4) 

points(bigmac$Price,Pred_pred$lwr,pch="o", col="green", lwd=4) 
points(bigmac$Price,Pred_pred$upr,pch="o", col="green", lwd=4) 



