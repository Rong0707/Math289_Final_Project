##log-loss: 0.477##
library(party)

train_target <- as.data.frame(fread("train_target.csv",stringsAsFactors=TRUE))
train_target <- as.factor(train_target[,1])
train_dat <- as.data.frame(fread("train_dat.csv",stringsAsFactors=TRUE))
test_dat <- as.data.frame(fread("test_dat.csv",stringsAsFactors=TRUE))

cat("Conditional Trees Model\n")
ctr = ctree(train_target ~ ., data=train_dat[,-1])
submission <- data.frame(Id = test_dat$ID)
a <- unlist(treeresponse(ctr,newdata = test_dat[,-1]),use.names = TRUE)
submission$PredictedProb <- a[seq(2,228786,2)]
write.csv(submission, file = "bnp_conditional_trees_submission.csv", row.names=FALSE)


