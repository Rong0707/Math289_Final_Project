## ------------------------------------------------------------------------
rm(list=ls(all=TRUE)) 

## ------------------------------------------------------------------------
library(xgboost)
library(data.table)
`%+%` <- function(a, b) paste(a, b, sep="")

## ------------------------------------------------------------------------
start_time <- Sys.time()
set.seed(2016289)

## ------------------------------------------------------------------------
message("Reading the train and test data")
train_raw <- fread("../data/train.csv",stringsAsFactors=FALSE, showProgress=FALSE, sep=",", na.strings = c("NA",""))
train_target <- train_raw$target
train_dat <- as.data.frame(train_raw[,-c(1,2),with = FALSE])

## ------------------------------------------------------------------------
train_dat <- train_dat[,names(train_dat)!="v22"] #too many levels
options(na.action='na.pass')
train_dat <- model.matrix(~.-1,train_dat)
options(na.action='na.omit')

## ------------------------------------------------------------------------
train_sub_id <- sample(nrow(train_dat),floor(nrow(train_dat)/5),replace = FALSE)
train_sub <- train_dat[-train_sub_id,]
test_sub <- train_dat[train_sub_id,]
train_sub_y <- train_target[-train_sub_id]
test_sub_y <- train_target[train_sub_id]

## ------------------------------------------------------------------------
xgtrain = xgb.DMatrix(data.matrix(train_sub), label = train_sub_y, missing = NA)
xgtest = xgb.DMatrix(data.matrix(test_sub), missing = NA)

## ------------------------------------------------------------------------
message("Data loaded")
print(difftime( Sys.time(), start_time, units = 'sec'))

## ------------------------------------------------------------------------
message("Training XGBoost classifiers")

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "logloss"
  , "eta" = 0.05
  , "min_child_weight" = 1
  , "max_depth" = 10
  , "nthread" = 16
  #, "scale_pos_weight" = 1
  #, "gamma" = 0.3
  #, "alpha" = 1e-05
  #, "lambda" = 0.1
  , "colsample_bytree" = 0.8
  , "subsample" = 0.8
)

#number of ensmeble models
n = 10

message("\nCross validate XGBoost classifier")
print(difftime( Sys.time(), start_time, units = 'sec'))

ensemble_cv <- xgb.cv(data = xgtrain, params = param0, nround = 5000, nfold = 2, early.stop.round = 10, print.every.n = 20)
best <- min(ensemble_cv$test.logloss.mean)
bestIter <- which(ensemble_cv$test.logloss.mean==best)

ensemble_p <- matrix(0, nrow = nrow(test_sub), ncol = n)
ensemble_model <- vector("list", n) 
watchlist <- list('train' = xgtrain)

for (i in 1:n){
  set.seed(2016 + 289*i)
  message("\nTraining the "%+%i%+%"th XGBoost classifier")
  print(difftime( Sys.time(), start_time, units = 'sec'))
  ensemble_model[[i]] <- xgb.train(data = xgtrain, params = param0, nround = round(bestIter*1.5), watchlist = watchlist, print.every.n = 20)
  ensemble_p[,i] <- predict(ensemble_model[[i]], xgtest)
}

## ------------------------------------------------------------------------
save(ensemble_p, ensemble_model, ensemble_cv, test_sub_y, file = "bnp-xgb-ohe-validation.RData")
