require(rminer)
rm(list = ls())
# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# standardize data to a zero mean and one standard deviation
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
H_red = holdout(1:1599, ratio = 2/3, internalsplit = TRUE)
red_wines_train = red_wines[H_red$tr,]
red_wines_test = red_wines[H_red$ts,]

white_wines[,(1:11)] <- scale(white_wines[,(1:11)])
H_white = holdout(1:4898, ratio = 2/3, internalsplit = TRUE, seed = 12345)
white_wines_train = white_wines[H_white$tr,]
white_wines_test = white_wines[H_white$ts,]


# create results table
results <- matrix(data = NA, nrow = 5, ncol = 6)
colnames(results) <- c('rw_MR', 'rw_NN', 'rw_SVM', 'ww_MR', 'ww_NN', 'ww_SVM')
rownames(results) <- c('MAE', 'Accuracy_0.25', 'Accuracy_0.5', 'Accuracy_1', 'Kappa_0.5')

# custom repeated k-fold cross validation function
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

# custom function to store predictions for graphing REC curves
store_predictions <- function(model, runs, folds, nobs, training, test){
  
  prediction_v <- vector("numeric", nobs)
  prediction_m <- matrix(data = NA, nrow = runs, ncol = nobs)
  for (i in 1:runs){
    model_reg <- fit(quality ~ ., data = training, method = c("kfold", folds, 12345), model = model, task = "reg")
    prediction_m[i,] <- predict(model_reg, test)
  }
  
  for (i in 1:nobs){
    prediction_v[i] <- mean(prediction_m[,i])
  }
  
  prediction_v
}

################################ Multiple Regression ################################ 
results <- custom_kfold(results, "mr", 20, 5, 1, 4)
reds_MR <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "mr", task = "reg")
whites_MR <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "mr", task = "reg")

################################ Neural Net ################################ 
results <- custom_kfold(results, "mlp", 20, 5, 2, 5)
reds_nnet <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "mlp", task = "reg")
whites_nnet <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "mlp", task = "reg")

################################  SVM ################################ 
results <- custom_kfold(results, "svm", 20, 5, 3, 6)
reds_SVM <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "svm", task = "reg")
whites_SVM <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "svm", task = "reg")

################################  Figures ################################

# histograms of quality
hist(red_wines$quality)
hist(white_wines$quality)

# table of results
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
