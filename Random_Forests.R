##log-loss: 0.46990 ##
library(ggplot2)
library(randomForest)
library(data.table)
library(mice)

train_target <- as.data.frame(fread("train_target.csv",stringsAsFactors=TRUE))
train_target <- as.factor(train_target[,1])
train_dat <- as.data.frame(fread("train_dat.csv",stringsAsFactors=TRUE))
test_dat <- as.data.frame(fread("test_dat.csv",stringsAsFactors=TRUE))

cat("Random Forests Model\n")
rf <- randomForest(x = train_dat[,-1], y = train_target, ntree=50, importance=TRUE)
submission <- data.frame(Id = test_dat$ID)
submission$PredictedProb <- predict(rf,test_dat[,-1],'prob')[,2]
write.csv(submission, file = "bnp_random_forest_r_submission.csv", row.names=FALSE)




