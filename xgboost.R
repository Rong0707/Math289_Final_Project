## ------------------------------------------------------------------------
rm(list=ls(all=TRUE)) 

## ------------------------------------------------------------------------
library(xgboost)
library(data.table)
library(pheatmap)
library(RColorBrewer)
library(ggplot2)
library(DiagrammeR)

## ------------------------------------------------------------------------
start_time <- Sys.time()
set.seed(2016289)

## ------------------------------------------------------------------------
cat("reading the train and test data\n")
train_raw <- fread("../data/train.csv",stringsAsFactors=TRUE)
train_target <- train_raw$target
train_dat <- as.data.frame(train_raw[,-c(1,2),with = FALSE])
test_raw <- fread("../data/test.csv",stringsAsFactors=TRUE)
test_dat <- as.data.frame(test_raw[,-1,with = FALSE])

## ------------------------------------------------------------------------
all_data <- rbind(train_dat,test_dat)
feature.names <- names(all_data)
for (f in feature.names) {
  if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
    all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}

## ------------------------------------------------------------------------
cormat <- cor(all_data, use="pairwise.complete.obs")
#heatmap <- pheatmap(cormat,show_rownames = F)
#d <- as.dist((1 - cormat)/2)
#plot(hclust(d))
#abline(h=0.1,col='red')
#group <- cutree(hclust(d), k = NULL, h = 0.05)
#sample_list <- NULL
#for (i in 1:max(group)){
#  sample_list <- c(sample_list,sample(names(group)[group==i],1))
#}
#all_data <- all_data[,sample_list]

## ------------------------------------------------------------------------
xgtrain = xgb.DMatrix(data.matrix(all_data[1:nrow(train_dat),]), label = train_target, missing = NA)
xgtest = xgb.DMatrix(data.matrix(all_data[(nrow(train_dat)+1):nrow(all_data),]), missing = NA)

## ------------------------------------------------------------------------
cat("Data read ")
print(difftime( Sys.time(), start_time, units = 'sec'))

## ------------------------------------------------------------------------
print( difftime( Sys.time(), start_time, units = 'sec'))
cat("Training a XGBoost classifier with cross-validation\n")
param0 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic"
        , "eval_metric" = "logloss"
        , "eta" = 0.05
        , "subsample" = 0.8
        , "colsample_bytree" = 1
        , "min_child_weight" = 1
        , "max_depth" = 10
        , "nthread" = 16
        #, "max_delta_step" = 1
        )
cv.res <- xgb.cv(data = xgtrain, params = param0, nround = 500, nfold = 3, early.stop.round = 10, print.every.n = 10, prediction = TRUE)
print( difftime( Sys.time(), start_time, units = 'sec'))

## ------------------------------------------------------------------------
#dat <- data.frame(round=as.integer(c(rownames(cv.res$dt),rownames(cv.res$dt))),method=c(rep('train',nrow(cv.res$dt)),rep('test',nrow(cv.res$dt))), logloss=c(cv.res$dt$train.logloss.mean,cv.res$dt$test.logloss.mean), se=c(cv.res$dt$train.logloss.std,cv.res$dt$test.logloss.std))
#limits <- aes(ymax = logloss + se, ymin = logloss - se)
#ggplot(dat,aes(round,logloss,color=method)) + geom_line() + geom_errorbar(limits,position = "dodge")

## ------------------------------------------------------------------------
best <- min(cv.res$dt$test.logloss.mean)
bestIter <- which(cv.res$dt$test.logloss.mean==best)
watchlist <- list('train' = xgtrain)
model <- xgb.train(data = xgtrain, params = param0, nround = bestIter*1.3, watchlist = watchlist, print.every.n = 20)

## ------------------------------------------------------------------------
cat("Making predictions\n")
p <- predict(model, xgtest)
submission <- data.frame(ID=test_raw$ID,PredictedProb=p)
write.csv(submission,"bnp-xgb-cv3.csv",row.names=F, quote=F)
print( difftime( Sys.time(), start_time, units = 'min'))

## ------------------------------------------------------------------------
importance <- xgb.importance(model = model)

## ------------------------------------------------------------------------
train_sub_id <- sample(nrow(train_dat),floor(nrow(train_dat)/5),replace = FALSE)
train_sub <- train_dat[-train_sub_id,]
test_sub <- train_dat[train_sub_id,]
train_sub_y <- train_target[-train_sub_id]
test_sub_y <- train_target[train_sub_id]
xgtrain_sub = xgb.DMatrix(data.matrix(train_sub), label = train_sub_y, missing = NA)
xgtest_sub = xgb.DMatrix(data.matrix(test_sub), missing = NA)
watchlist_sub <- list('train' = xgtrain_sub)
model_sub <- xgb.train(data = xgtrain_sub, params = param0, nround = bestIter*1.2, watchlist = watchlist_sub, print.every.n = 20)
p_sub <- predict(model_sub, xgtest_sub)

## ------------------------------------------------------------------------
save(cv.res, model, importance, submission, cormat, model_sub, test_sub_y, p_sub, file = "bnp-xgb-cv3.RData")


