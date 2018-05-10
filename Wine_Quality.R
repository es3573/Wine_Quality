################################ Summary ################################
# This file is solely for the reproduction of results in the study. It covers the 
# multiple linear regression, multilayer perceptron, and kernel SVM models in
# a regression approach to modeling the Wine data sets. 

require(rminer)
rm(list = ls())
# Import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

################################ Preprocessing ################################ 
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
H_red = holdout(1:1599, ratio = 2/3, internalsplit = TRUE, seed = 12345)
red_wines_train = red_wines[H_red$tr,]
red_wines_test = red_wines[H_red$ts,]

white_wines[,(1:11)] <- scale(white_wines[,(1:11)])
H_white = holdout(1:4898, ratio = 2/3, internalsplit = TRUE, seed = 12345)
white_wines_train = white_wines[H_white$tr,]
white_wines_test = white_wines[H_white$ts,]

################################ Custom Functions ################################ 
# Function for k-fold cross validation with repeated runs and calculating MAE, ACC, KAPPA values in regression setting
# specialized for Rminer functions according to study
custom_kfold <- function(results_m, model, runs, folds, red_index, white_index){
  
  red_MAE <- vector("numeric", runs)
  red_0.25 <- vector("numeric", runs)
  red_0.5 <- vector("numeric", runs)
  red_1 <- vector("numeric", runs)
  red_kappa <- vector("numeric", runs)
  white_MAE <- vector("numeric", runs)
  white_0.25 <- vector("numeric", runs)
  white_0.5 <- vector("numeric", runs)
  white_1 <- vector("numeric", runs)
  white_kappa <- vector("numeric", runs)
  
  for (i in 1:runs){
    reds_reg <- fit(quality ~ ., data = red_wines_train, method = c("kfold", folds, 12345), model = model, task = "reg")
    whites_reg <- fit(quality ~ ., data = white_wines_train, method = c("kfold", folds, 12345), model = model, task = "reg")
    
    P_red_reg = predict(reds_reg, red_wines_test)
    P_white_reg = predict(whites_reg, white_wines_test)
    
    red_MAE[i] <- mmetric(P_red_reg, red_wines_test$quality, metric = "MAE")
    red_0.25[i] <- mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.25)
    red_0.5[i] <- mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.5)
    red_1[i] <- mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 1)
    red_kappa[i] <- mmetric(factor(round(P_red_reg)), red_wines_test$quality, metric = "KAPPA", val = 0.5)
    white_MAE[i] <- mmetric(P_white_reg, white_wines_test$quality, metric = "MAE")
    white_0.25[i] <- mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.25)
    white_0.5[i] <- mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.5)
    white_1[i] <- mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 1)
    white_kappa[i] <- mmetric(factor(round(P_white_reg)), white_wines_test$quality, metric = "KAPPA", val = 0.5)
  }
  
  results_m[1, red_index] <- round(mean(red_MAE), 2)
  results_m[2, red_index] <- 100 * round(mean(red_0.25),3)
  results_m[3, red_index] <- 100 * round(mean(red_0.5), 3)
  results_m[4, red_index] <- 100 * round(mean(red_1), 3)
  results_m[5, red_index] <- round(mean(red_kappa),1)
  results_m[1, white_index] <- round(mean(white_MAE), 2) 
  results_m[2, white_index] <- 100 * round(mean(white_0.25), 3)
  results_m[3, white_index] <- 100 * round(mean(white_0.5), 3)
  results_m[4, white_index] <- 100 * round(mean(white_1), 3)
  results_m[5, white_index] <- round(mean(white_kappa),1)
  
  results_m
}

# Function to store predictions 
store_predictions <- function(model, runs, folds, n_obs, training, test){
  
  prediction_v <- vector("numeric", n_obs)
  prediction_m <- matrix(data = NA, nrow = runs, ncol = n_obs)
  for (i in 1:runs){
    model_reg <- fit(quality ~ ., data = training, method = c("kfold", folds, 12345), model = model, task = "reg")
    prediction_m[i,] <- predict(model_reg, test)
  }
  
  for (i in 1:n_obs){
    prediction_v[i] <- mean(prediction_m[,i])
  }
  
  prediction_v
}

# Function for assigning class based off procedure in study
class_assignment <- function(truth, regression, tolerance){
  prediction <- vector("numeric", length(truth))
  for (i in 1:length(truth)){
    
    if (abs(truth[i] - regression[i]) <= tolerance){
      prediction[i] <- truth[i]
    } else{
      prediction[i] <- round(regression[i])
    }
    
  }
  prediction
}

# Function for creating confusion matrix and calculating precision
confusion <- function(predictions, truth){
  n_truth_classes <- length(unique(truth))
  n_pred_classes <- length(unique(predictions))
  confusion_m <- matrix(data = 0L, nrow = n_truth_classes + 1, ncol = n_pred_classes)
  min_truth <- min(unique(truth))
  min_pred <- min(unique(predictions))
  
  # Create confusion matrix
  for (k in 1:length(predictions)){
    col_ind <- predictions[k] - (min_pred - 1) 
    row_ind <- truth[k] - (min_truth - 1)
    confusion_m[row_ind,col_ind] <- confusion_m[row_ind, col_ind] + 1
  }
  
  # Set starting diagonal point to find correct matches
  if (min_pred > min_truth){
    diag_row <- 1 + (min_pred - min_truth)
    diag_col <- 1
  } else if (min_pred < min_truth){
    diag_row <- 1
    diag_col <- 1 + (min_truth - min_pred)
  } else{
    diag_row <- 1
    diag_col <- 1
  }
  
  # Calculate precision of predictions
  for (i in 1:n_pred_classes){
    if (diag_row <= n_truth_classes && diag_col <= n_pred_classes){
      confusion_m[n_truth_classes + 1, diag_col] <- 100 * round(confusion_m[diag_row, diag_col] / sum(confusion_m[,i]),3)
      diag_row <- diag_row + 1
      diag_col <- diag_col + 1
    }
  }
  
  confusion_m
}

################################ Comparison of models ################################ 
# Create results table
results <- matrix(data = NA, nrow = 7, ncol = 6)
colnames(results) <- c('rw_MR', 'rw_NN', 'rw_SVM', 'ww_MR', 'ww_NN', 'ww_SVM')
rownames(results) <- c('MAE', 'Accuracy_0.25', 'Accuracy_0.5', 'Accuracy_1', 'Kappa_0.5', 'inputs', 'model')

################################ Multiple Regression ################################ 
results <- custom_kfold(results, "mr", 20, 5, 1, 4)

red_MR_reg <- mining(quality ~ ., data = red_wines,Runs=20,method=c("holdout",2/3,12345),model="mr", task = "reg",
                  search=list(search=mparheuristic("ksvm",lower = -2, upper = 7, by = 1),method=c("kfold",5,123),metric="MAE"),
                  feature="sabsv")

white_MR_reg <- mining(quality ~ ., data = white_wines,Runs=20,method=c("holdout",2/3,12345),model="mr", task = "reg",
                    search=list(search=mparheuristic("ksvm",lower = -2, upper = 7, by = 1),method=c("kfold",5,123),metric="MAE"),
                    feature="sabsv")

# Calculate number of inputs
red_MR_feats <- 0
for (i in 1:20){red_MR_feats = red_MR_feats +  length(red_MR_reg[["attributes"]][[i]])}
red_MR_input <- red_MR_feats/20
results[6,1] <- red_MR_input

white_MR_feats <- 0
for (i in 1:20){white_MR_feats = white_MR_feats +  length(white_MR_reg[["attributes"]][[i]])}
white_MR_input <- white_MR_feats/20
results[6,4] <- white_MR_input

################################ Neural Net ################################ 
results <- custom_kfold(results, "mlp", 20, 5, 2, 5)

red_nnet_reg <- mining(quality ~ ., data = red_wines,Runs=20,method=c("holdout",2/3,12345),model="mlp", task = "reg",
                    search=list(search=mparheuristic("ksvm",lower = -2, upper = 7, by = 1),method=c("kfold",5,123),metric="MAE"),
                    feature="sabsv")

white_nnet_reg <- mining(quality ~ ., data = white_wines,Runs=20,method=c("holdout",2/3,12345),model="mlp", task = "reg",
                      search=list(search=mparheuristic("ksvm",lower = -2, upper = 7, by = 1),method=c("kfold",5,123),metric="MAE"),
                      feature="sabsv")

# Calculate number of inputs
red_nnet_feats <- 0
for (i in 1:20){red_nnet_feats = red_nnet_feats +  length(red_nnet_reg[["attributes"]][[i]])}
red_nnet_input <- red_nnet_feats/20
results[6,2] <- red_nnet_input

white_nnet_feats <- 0
for (i in 1:20){white_nnet_feats = white_nnet_feats +  length(white_nnet_reg[["attributes"]][[i]])}
white_nnet_input <- white_nnet_feats/20
results[6,5] <- white_nnet_input


################################  SVM ################################ 
results <- custom_kfold(results, "svm", 20, 5, 3, 6)

red_SVM_reg <- mining(quality ~ ., data = red_wines,Runs = 20,method = c("holdout",2/3,12345),model = "svm", task = "reg",
                   search=list(search = mparheuristic("ksvm",lower = -2, upper = 7, by = 1),method = c("kfold",5,123),metric = "MAE"),
                   feature = "sabsv")

white_SVM_reg <- mining(quality ~ ., data = white_wines,Runs=20,method = c("holdout",2/3,12345),model = "svm", task = "reg",
                     search = list(search = mparheuristic("ksvm",lower = -2, upper = 7, by = 1),method = c("kfold",5,123),metric = "MAE"),
                     feature = "sabsv")

# Calculate number of inputs
red_SVM_feats <- 0
for (i in 1:20){red_SVM_feats = red_SVM_feats +  length(red_SVM_reg$attributes[[i]])}
red_SVM_input <- red_SVM_feats/20
results[6,3] <- round(red_SVM_input,2)

white_SVM_feats <- 0
for (i in 1:20){white_SVM_feats = white_SVM_feats +  length(white_SVM_reg[["attributes"]][[i]])}
white_SVM_input <- white_SVM_feats/20
results[6,6] <- white_SVM_input

# Compute sigma value for SVM
red_SVM_sigma <- 0
for (i in 1:20){red_SVM_sigma = red_SVM_sigma + red_SVM_reg$mpar[[i]]$kpar$sigma}
red_SVM_gamma <- 1/(2*(red_SVM_sigma/20)**2)
results[7,3] <- round(red_SVM_gamma,2)

white_SVM_sigma <- 0
for (i in 1:20){white_SVM_sigma = white_SVM_sigma + white_SVM_reg$mpar[[i]]$kpar$sigma}
white_SVM_gamma <- 1/(2*(white_SVM_sigma/20)**2)
results[7,6] <- round(white_SVM_gamma,2)

################################  Figures ################################

# Histograms of quality
hist(red_wines$quality)
hist(white_wines$quality)

# Table of results
print(results.table <- as.table(results))

# REC curves
P_red_MR <- store_predictions("mr", 20, 5, length(red_wines_test[,1]), red_wines_train, red_wines_test)
P_white_MR <- store_predictions("mr", 20, 5, length(white_wines_test[,1]), white_wines_train, white_wines_test)
P_red_nnet <- store_predictions("mlp", 20, 5, length(red_wines_test[,1]), red_wines_train, red_wines_test)
P_white_nnet <- store_predictions("mlp", 20, 5, length(white_wines_test[,1]), white_wines_train, white_wines_test)
P_red_SVM <- store_predictions("svm", 20, 5, length(red_wines_test[,1]), red_wines_train, red_wines_test)
P_white_SVM <- store_predictions("svm", 20, 5, length(white_wines_test[,1]), white_wines_train, white_wines_test)

L_red = vector("list", 3)
pred_r = vector("list", 1)
test_r = vector("list", 1)

L_white = vector("list", 3)
pred_w = vector("list", 1)
test_w = vector("list", 1)

pred_r[[1]] = red_wines_test$quality
test_r[[1]] = P_red_SVM
L_red[[1]] = list(pred = pred_r, test = test_r, runs = 1)
test_r[[1]] = P_red_nnet
L_red[[2]] = list(pred = pred_r, test = test_r, runs = 1)
test_r[[1]] = P_red_MR
L_red[[3]] = list(pred = pred_r, test = test_r, runs = 1)

pred_w[[1]] = white_wines_test$quality
test_w[[1]] = P_white_SVM
L_white[[1]] = list(pred = pred_w, test = test_w, runs = 1)
test_w[[1]] = P_white_nnet
L_white[[2]] = list(pred = pred_w, test = test_w, runs = 1)
test_w[[1]] = P_white_MR
L_white[[3]] = list(pred = pred_w, test = test_w, runs = 1)

mgraph(L_red, graph = "REC", xval = 2, leg = c("SVM", "NN", "MR"), main = "Red wine")
mgraph(L_white, graph = "REC", xval = 2, leg = c("SVM", "NN", "MR"), main = "White wine")

# Bar plot of importance of features
red_imp  = sort(colMeans(red_SVM_reg$sen)[1:11])
red_bp = barplot(red_imp, main="Red Wine", horiz=TRUE, col = 'white')
text(x=red_imp, y = red_bp, labels=colnames(red_wines)[order(colMeans(red_SVM_reg$sen))][2:12], cex = 0.6, xpd = TRUE, pos = 2)

white_imp  = sort(colMeans(white_SVM_reg$sen)[1:11])
white_bp = barplot(white_imp, main="white Wine", horiz=TRUE, col = 'white')
text(x=white_imp, y = white_bp, labels=colnames(white_wines)[order(colMeans(white_SVM_reg$sen))][2:12], cex = 0.6, xpd = TRUE, pos = 2)


# Confusion matrices
P_red_SVM <- store_predictions("ksvm", 20, 5, length(red_wines[,1]), red_wines_train, red_wines)
P_red_SVM_c <- class_assignment(red_wines$quality, P_red_SVM, 0.5)
confusion_red_0.5 <- confusion(P_red_SVM_c, red_wines$quality)
rownames(confusion_red_0.5) <- c(min(unique(red_wines$quality)):max(unique(red_wines$quality)), 'Precision')
colnames(confusion_red_0.5) <- c(min(unique(P_red_SVM_c)):max(unique(P_red_SVM_c)))
print(as.table(confusion_red_0.5))

P_red_SVM_c <- class_assignment(red_wines$quality, P_red_SVM, 1)
confusion_red_1 <- confusion(P_red_SVM_c, red_wines$quality)
rownames(confusion_red_1) <- c(min(unique(red_wines$quality)):max(unique(red_wines$quality)), 'Precision')
colnames(confusion_red_1) <- c(min(unique(P_red_SVM_c)):max(unique(P_red_SVM_c)))
print(as.table(confusion_red_1))

P_white_SVM <- store_predictions("svm", 20, 5, length(white_wines[,1]), white_wines_train, white_wines)
P_white_SVM_c <- class_assignment(white_wines$quality, P_white_SVM, 0.5)
confusion_white_0.5 <- confusion(P_white_SVM_c, white_wines$quality)
rownames(confusion_white_0.5) <- c(min(unique(white_wines$quality)):max(unique(white_wines$quality)), 'Precision')
colnames(confusion_white_0.5) <- c(min(unique(P_white_SVM_c)):max(unique(P_white_SVM_c)))
print(as.table(confusion_white_0.5))

P_white_SVM_c <- class_assignment(white_wines$quality, P_white_SVM, 1)
confusion_white_1 <- confusion(P_white_SVM_c, white_wines$quality)
rownames(confusion_white_1) <- c(min(unique(white_wines$quality)):max(unique(white_wines$quality)), 'Precision')
colnames(confusion_white_1) <- c(min(unique(P_white_SVM_c)):max(unique(P_white_SVM_c)))
print(as.table(confusion_white_1))







