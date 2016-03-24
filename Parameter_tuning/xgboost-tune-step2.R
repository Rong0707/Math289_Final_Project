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
  , "nthread" = 8
  , "scale_pos_weight" = 1
  , "subsample" = 0.8
  , "colsample_bytree" = 0.8
  , "max_depth" = 6
  , "min_child_weight" = 3
)

gamma = seq(0,0.5,0.1)

total_n <- length(gamma)
ensemble_cv <- vector("list", total_n)

for (i in 1:length(gamma)){
    message("\nCross validate the "%+%i%+%"th XGBoost classifier")
    print(difftime( Sys.time(), start_time, units = 'sec'))
    ensemble_cv[[i]] <- xgb.cv(data = xgtrain, params = param0, gamma = gamma[i], nround = 500, nfold = 2, early.stop.round = 10, print.every.n = 20)
}

## ------------------------------------------------------------------------
save(ensemble_cv, gamma, file = "bnp-xgb-tune-step2.RData")

cv_sum <- data.frame(test.logloss=sapply(ensemble_cv,function (x) min(x$test.logloss.mean)), gamma = gamma)
pdf("xgboost-tune-step2.pdf")
ggplot(cv_sum, aes(gamma, test.logloss)) + geom_line()
dev.off()


