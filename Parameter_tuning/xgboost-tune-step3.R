## ------------------------------------------------------------------------
rm(list=ls(all=TRUE)) 

## ------------------------------------------------------------------------
library(xgboost)
library(data.table)
library(ggplot2)
`%+%` <- function(a, b) paste(a, b, sep="")

## ------------------------------------------------------------------------
start_time <- Sys.time()
set.seed(2016289)

## ------------------------------------------------------------------------
message("Reading the train and test data")
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
xgtrain = xgb.DMatrix(data.matrix(all_data[1:nrow(train_dat),]), label = train_target, missing = NA)
xgtest = xgb.DMatrix(data.matrix(all_data[(nrow(train_dat)+1):nrow(all_data),]), missing = NA)

## ------------------------------------------------------------------------
message("Data loaded")
print(difftime( Sys.time(), start_time, units = 'sec'))

## ------------------------------------------------------------------------
message("Training XGBoost classifiers")

param0 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic"
        , "eval_metric" = "logloss"
        , "eta" = 0.1
        , "min_child_weight" = 3
        , "max_depth" = 6
        , "nthread" = 16
        , "gamma" = 0.3
        , "scale_pos_weight" = 1
        )

colsample_bytree = seq(0.6,1,0.1)
subsample = seq(0.6,1,0.1)

total_n <- length(colsample_bytree)*length(subsample)
ensemble_cv <- vector("list", total_n)

for (i in 1:length(colsample_bytree)){
  for (j in 1:length(subsample)){
    n <- (i-1)*length(subsample)+j
    message("\nCross validate the "%+%n%+%"th XGBoost classifier")
    print(difftime( Sys.time(), start_time, units = 'sec'))
    ensemble_cv[[n]] <- xgb.cv(data = xgtrain, params = param0, colsample_bytree = colsample_bytree[i], subsample = subsample[j], nround = 500, nfold = 2, early.stop.round = 10, print.every.n = 20)
  }
}

## ------------------------------------------------------------------------
save(ensemble_cv, colsample_bytree, subsample, file = "bnp-xgb-tune-step3.RData")

cv_sum <- data.frame(test.logloss=sapply(ensemble_cv,function (x) min(x$test.logloss.mean)), colsample_bytree=rep(colsample_bytree, each = length(subsample)), subsample=rep(subsample,length(colsample_bytree)))
pdf("xgboost-tune-step3.pdf")
ggplot(cv_sum, aes(factor(colsample_bytree), factor(subsample))) + geom_tile(aes(fill = test.logloss), colour = "white") + scale_fill_gradient(low = "white",high = "steelblue")
dev.off()

cv_sum[min(cv_sum$test.logloss)==cv_sum$test.logloss,]
