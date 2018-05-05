require(rminer)
rm(list = ls())
# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# tabulate data
hist(red_wines$quality)
hist(white_wines$quality)

# standardize data to a zero mean and one standard deviation
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
H_red = holdout(1:1599, ratio = 2/3, internalsplit = TRUE)
red_wines_train = red_wines[H_red$tr,]
red_wines_test = red_wines[H_red$ts,]

white_wines[,(1:11)] <- scale(white_wines[,(1:11)])
H_white = holdout(1:4898, ratio = 2/3, internalsplit = TRUE)
white_wines_train = white_wines[H_white$tr,]
white_wines_test = white_wines[H_white$ts,]


# create results table
results <- matrix(data = NA, nrow = 4, ncol = 6)
colnames(results) <- c('rw_MR', 'rw_NN', 'rw_SVM', 'ww_MR', 'ww_NN', 'ww_SVM')
rownames(results) <- c('MAE', 'Accuracy_0.25', 'Accuracy_0.5', 'Accuracy_1')

# custom repeated k-fold cross validation function
custom_kfold <- function(results_m, model, runs, folds, red_index, white_index){
  
  red_MAE <- vector("numeric", runs)
  red_0.25 <- vector("numeric", runs)
  red_0.5 <- vector("numeric", runs)
  red_1 <- vector("numeric", runs)
  white_MAE <- vector("numeric", runs)
  white_0.25 <- vector("numeric", runs)
  white_0.5 <- vector("numeric", runs)
  white_1 <- vector("numeric", runs)
  
  for (i in 1:runs){
    reds_reg <- fit(quality ~ ., data = red_wines_train, method = c("kfold", folds, 123), model = model, task = "reg")
    whites_reg <- fit(quality ~ ., data = white_wines_train, method = c("kfold", folds, 123), model = model, task = "reg")
    
    P_red_reg = predict(reds_reg, red_wines_test)
    P_white_reg = predict(whites_reg, white_wines_test)
    
    red_MAE[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "MAE"))
    red_0.25[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.25))
    red_0.5[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.5))
    red_1[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 1))
    white_MAE[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "MAE"))
    white_0.25[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.25))
    white_0.5[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.5))
    white_1[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 1))
  }
  
  results_m[1,red_index]<- round(mean(red_MAE), 2)
  results_m[2,red_index]<- 100 * round(mean(red_0.25),3)
  results_m[3,red_index]<- 100 * round(mean(red_0.5), 3)
  results_m[4,red_index]<- 100 * round(mean(red_1), 3)
  results_m[1,white_index]<- round(mean(white_MAE), 2) 
  results_m[2,white_index]<- 100 * round(mean(white_0.25), 3)
  results_m[3,white_index]<- 100 * round(mean(white_0.5), 3)
  results_m[4,white_index]<- 100 * round(mean(white_1), 3)
  
  results_m
}


################################ Multiple Regression ################################ 
results <- custom_kfold(results, "mr", 20, 5, 1, 4)
#reds_MR <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "mr", task = "reg")
#whites_MR <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "mr", task = "reg")

################################ Neural Net ################################ 
results <- custom_kfold(results, "mlp", 20, 5, 2, 5)
#reds_nnet <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "mlp", task = "reg")
#whites_nnet <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "mlp", task = "reg")

################################  SVM ################################ 
results <- custom_kfold(results, "svm", 20, 5, 3, 6)
#reds_SVM <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "svm", task = "reg")
#whites_SVM <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "svm", task = "reg")

##################################################################################################
print(results.table <- as.table(results))
