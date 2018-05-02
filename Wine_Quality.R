rm(list = ls())
# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# standardize data to a zero mean and one standard deviation

red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
white_wines[,(1:11)] <- scale(white_wines[,(1:11)])


# 5-fold cross-validation with 20 runs
require(caret)
train_control <-trainControl(method = "repeatedcv", number = 5, repeats = 20, verboseIter = TRUE)

# Multiple Regression 
reds_MR <- train(quality ~ ., data = red_wines, method = "lm", trControl = train_control)
whites_MR <- train(quality ~ ., data = white_wines, method = "lm", trControl = train_control)

# nnet 
reds_nnet <-train(quality ~ ., data = red_wines, method = "nnet", trControl = train_control)
whites_nnet <-train(quality ~ . , data = white_wines, method = "nnet", trControl = train_control)