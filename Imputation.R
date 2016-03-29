library(ggplot2)
library(randomForest)
library(data.table)
library(mice)

##variables that have more than 30% missing data are meaningless in the model, so I just deleted them and used 30 variables.
set.seed(1)
train_raw <- fread("train.csv",stringsAsFactors=TRUE)
train_raw <- as.data.frame(train_raw[,c(1,2,5,12,14,16,23,24,26,32,33,36,40,42,49,52,54,58,64,68,73,74,76,77,81,93,109,112,114,115,127,131),with = FALSE])
train_target <- train_raw$target
write.csv(train_target, file = "train_target.csv", row.names=FALSE)
train_raw <- train_raw[-2]

set.seed(1)
test_raw <- fread("test.csv",stringsAsFactors=TRUE)
test_raw <- as.data.frame(test_raw[,c(1,4,11,13,15,22,23,25,31,32,35,39,41,48,51,53,57,63,67,72,73,75,76,80,92,108,111,113,114,126,130),with = FALSE])

all_data <- rbind(train_raw,test_raw)
feature.names <- names(all_data)
for (f in feature.names) {
  if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
    all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}

train_raw <- all_data[1:nrow(train_raw),]
test_raw <- all_data[(nrow(train_raw)+1):nrow(all_data),]

imp <- mice(train_raw,m = 1, method = "pmm")
train_dat <- complete(imp,1)
write.csv(train_dat, file = "train_dat.csv", row.names=FALSE)

imp <- mice(test_raw,m = 1, method = "pmm")
test_dat <- complete(imp,1)
write.csv(test_dat, file = "test_dat.csv", row.names=FALSE)


