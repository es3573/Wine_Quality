require(caret)
rm(list = ls())
# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# tabulate data
histogram(red_wines$quality)
histogram(white_wines$quality)

# standardize data to a zero mean and one standard deviation
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
white_wines[,(1:11)] <- scale(white_wines[,(1:11)])

# 5-fold cross-validation with 20 runs
train_control <-trainControl(method = "repeatedcv", number = 5, repeats = 20)

# Multiple Regression 
reds_MR <- train(quality ~ ., data = red_wines, method = "lm", trControl = train_control)
whites_MR <- train(quality ~ ., data = white_wines, method = "lm", trControl = train_control)

# nnet 
reds_nnet <-train(quality ~ ., data = red_wines, method = "nnet", trControl = train_control, linout = TRUE)
whites_nnet <-train(quality ~ . , data = white_wines, method = "nnet", trControl = train_control, linout = TRUE)

# Gaussian SVM
reds_SVM <-train(quality ~., data = red_wines, method = "svmRadial", trControl = train_control)
whites_SVM <-train(quality ~., data = white_wines, method = "svmRadial", trControl = train_control)

# MAE comparisons
reds_MR_MAE <- reds_MR$results$MAE
whites_MR_MAE <- whites_MR$results$MAE
reds_nnet_MAE <- mean(reds_nnet$results$MAE)
whites_nnet_MAE <- mean(whites_nnet$results$MAE)
reds_SVM_MAE <-mean(reds_SVM$results$MAE)
whites_SVM_MAE <-mean(whites_SVM$results$MAE)

# R_squared comparisons
reds_MR_R2 <-reds_MR$results$Rsquared
whites_MR_R2 <-whites_MR$results$Rsquared
reds_nnet_R2 <- mean(reds_nnet$results$Rsquared)
whites_nnet_R2 <- mean(whites_nnet$results$Rsquared)
reds_SVM_R2 <- mean(reds_SVM$results$Rsquared)
whites_SVM_R2 <- mean(whites_SVM$results$Rsquared)
