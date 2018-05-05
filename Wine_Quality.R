require(rminer)
rm(list = ls())
# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# tabulate data
histogram(red_wines$quality)
histogram(white_wines$quality)

# standardize data to a zero mean and one standard deviation
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
H_red = holdout(1:1599, ratio = 2/3, internalsplit = TRUE)
red_wines_train = red_wines[H_red$tr,]
red_wines_test = red_wines[H_red$ts,]

white_wines[,(1:11)] <- scale(white_wines[,(1:11)])
H_white = holdout(1:4898, ratio = 2/3, internalsplit = TRUE)
white_wines_train = white_wines[H_white$tr,]
white_wines_test = white_wines[H_white$ts,]

################################ Multiple Regression ################################ 
N <- 20
red_MR_MAE <- vector("numeric", N)
red_MR_0.25 <- vector("numeric", N)
red_MR_0.5 <- vector("numeric", N)
red_MR_1 <- vector("numeric", N)
white_MR_MAE <- vector("numeric", N)
white_MR_0.25 <- vector("numeric", N)
white_MR_0.5 <- vector("numeric", N)
white_MR_1 <- vector("numeric", N)

for (i in 1:N){
  reds_MR_reg <- fit(quality ~ ., data = red_wines_train, method = c("kfold", 5, 123), model = "mr", task = "reg")
  whites_MR_reg <- fit(quality ~ ., data = white_wines_train, method = c("kfold", 5, 123), model = "mr", task = "reg")
  
  P_red_reg = predict(reds_MR_reg, red_wines_test)
  P_white_reg = predict(whites_MR_reg, white_wines_test)
  
  red_MR_MAE[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "MAE"))
  red_MR_0.25[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.25))
  red_MR_0.5[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.5))
  red_MR_1[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 1))
  white_MR_MAE[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "MAE"))
  white_MR_0.25[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.25))
  white_MR_0.5[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.5))
  white_MR_1[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 1))
}

reds_MR <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "mr", task = "reg")
whites_MR <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "mr", task = "reg")
# print(mmetric(reds_MR, metric = "MAE"))
# print(mmetric(reds_MR, metric = "TOLERANCE", val = 0.25))
# print(mmetric(reds_MR, metric = "TOLERANCE", val = 0.5))
# print(mmetric(reds_MR, metric = "TOLERANCE", val = 1))
# print(mmetric(whites_MR, metric = "MAE"))
# print(mmetric(whites_MR, metric = "TOLERANCE", val = 0.25))
# print(mmetric(whites_MR, metric = "TOLERANCE", val = 0.5))
# print(mmetric(whites_MR, metric = "TOLERANCE", val = 1))


################################ Neural Net ################################ 
red_nnet_MAE <- vector("numeric", N)
red_nnet_0.25 <- vector("numeric", N)
red_nnet_0.5 <- vector("numeric", N)
red_nnet_1 <- vector("numeric", N)
white_nnet_MAE <- vector("numeric", N)
white_nnet_0.25 <- vector("numeric", N)
white_nnet_0.5 <- vector("numeric", N)
white_nnet_1 <- vector("numeric", N)

for (i in 1:N){
  reds_nnet_reg <- fit(quality ~ ., data = red_wines_train, method = c("kfold", 5, 123), model = "mlp", task = "reg")
  whites_nnet_reg <- fit(quality ~ ., data = white_wines_train, method = c("kfold", 5, 123), model = "mlp", task = "reg")
  
  P_red_reg = predict(reds_nnet_reg, red_wines_test)
  P_white_reg = predict(whites_nnet_reg, white_wines_test)
  
  red_nnet_MAE[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "MAE"))
  red_nnet_0.25[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.25))
  red_nnet_0.5[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.5))
  red_nnet_1[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 1))
  white_nnet_MAE[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "MAE"))
  white_nnet_0.25[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.25))
  white_nnet_0.5[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.5))
  white_nnet_1[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 1))
}


reds_nnet <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "mlp", task = "reg")
whites_nnet <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "mlp", task = "reg")
# print(mmetric(reds_nnet, metric = "MAE"))
# print(mmetric(reds_nnet, metric = "TOLERANCE", val = 0.25))
# print(mmetric(reds_nnet, metric = "TOLERANCE", val = 0.5))
# print(mmetric(reds_nnet, metric = "TOLERANCE", val = 1))
# print(mmetric(whites_nnet, metric = "MAE"))
# print(mmetric(whites_nnet, metric = "TOLERANCE", val = 0.25))
# print(mmetric(whites_nnet, metric = "TOLERANCE", val = 0.5))
# print(mmetric(whites_nnet, metric = "TOLERANCE", val = 1))

################################  SVM ################################ 
red_SVM_MAE <- vector("numeric", N)
red_SVM_0.25 <- vector("numeric", N)
red_SVM_0.5 <- vector("numeric", N)
red_SVM_1 <- vector("numeric", N)
white_SVM_MAE <- vector("numeric", N)
white_SVM_0.25 <- vector("numeric", N)
white_SVM_0.5 <- vector("numeric", N)
white_SVM_1 <- vector("numeric", N)

for (i in 1:N){
  reds_SVM_reg <- fit(quality ~ ., data = red_wines_train, method = c("kfold", 5, 123), model = "svm", task = "reg")
  whites_SVM_reg <- fit(quality ~ ., data = white_wines_train, method = c("kfold", 5, 123), model = "svm", task = "reg")
  
  P_red_reg = predict(reds_SVM_reg, red_wines_test)
  P_white_reg = predict(whites_SVM_reg, white_wines_test)
  
  red_SVM_MAE[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "MAE"))
  red_SVM_0.25[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.25))
  red_SVM_0.5[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 0.5))
  red_SVM_1[i]<-(mmetric(P_red_reg, red_wines_test$quality, metric = "TOLERANCE", val = 1))
  white_SVM_MAE[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "MAE"))
  white_SVM_0.25[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.25))
  white_SVM_0.5[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 0.5))
  white_SVM_1[i]<-(mmetric(P_white_reg, white_wines_test$quality, metric = "TOLERANCE", val = 1))
}

reds_SVM <- mining(quality ~ ., data = red_wines, runs = 20, method = c("kfold", 5, 123), model = "svm", task = "reg")
whites_SVM <- mining(quality ~ ., data = white_wines, runs = 20, method = c("kfold", 5, 123), model = "svm", task = "reg")
# print(mmetric(reds_SVM, metric = "MAE"))
# print(mmetric(reds_SVM, metric = "TOLERANCE", val = 0.25))
# print(mmetric(reds_SVM, metric = "TOLERANCE", val = 0.5))
# print(mmetric(reds_SVM, metric = "TOLERANCE", val = 1))
# print(mmetric(whites_SVM, metric ="MAE"))
# print(mmetric(whites_SVM, metric = "TOLERANCE", val = 0.25))
# print(mmetric(whites_SVM, metric = "TOLERANCE", val = 0.5))
# print(mmetric(whites_SVM, metric = "TOLERANCE", val = 1))
