require(xgboost)

require(rminer)
rm(list = ls())
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

# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# standardize data to a zero mean and one standard deviation
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
H_red = holdout(1:1599, ratio = 2/3, internalsplit = TRUE)
red_wines_train_val = red_wines[H_red$tr,]
red_wines_test = red_wines[H_red$ts,]
TR_red = holdout(1:1066, ratio = 0.8, internalsplit = TRUE)
red_wines_train = red_wines_train_val[TR_red$tr,]
red_wines_val = red_wines_train_val[TR_red$ts,]

white_wines[,(1:11)] <- scale(white_wines[,(1:11)])
H_white = holdout(1:4898, ratio = 2/3, internalsplit = TRUE, seed = 12345)
white_wines_train_val = white_wines[H_white$tr,]
white_wines_test = white_wines[H_white$ts,]
TR_white = holdout(1:3265, ratio = 0.8, internalsplit = TRUE)
white_wines_train = white_wines_train_val[TR_white$tr,]
white_wines_val = white_wines_train_val[TR_white$ts,]


# drop label for each data
red_x_train <- red_wines_train[,(1:11)]
red_x_test <- red_wines_test[,(1:11)]
red_x_val <- red_wines_val[,(1:11)]

white_x_train <- white_wines_train[,(1:11)]
white_x_test <- white_wines_test[,(1:11)]
white_x_val <- white_wines_val[,(1:11)]

red_x_train <- t(t(array(red_x_train)))
red_x_test <- t(t(array(red_x_test)))
red_x_val <- t(t(array(red_x_val)))

white_x_train <- t(t(array(white_x_train)))
white_x_test <- t(t(array(white_x_test)))
white_x_val <- t(t(array(white_x_val)))

red_y_train <- red_wines_train$quality
red_y_test <- red_wines_test$quality
red_y_val <- red_wines_val$quality

white_y_train <- white_wines_train$quality
white_y_test <- white_wines_test$quality
white_y_val <- white_wines_val$quality

################## red ##################
red_boost <- xgboost(data = red_x_train, label = red_y_train-3, max.depth = 2, eta = .01,
                     nthread = 2, nround = 16000, objective = "multi:softmax", num_class = 6)

# validation predict
red_val_pred <- predict(red_boost, red_x_val)

# validation accuracy
red_val_acc <- mean(red_val_pred == red_y_val-3)
print(red_val_acc)

# test predict
red_pred <- predict(red_boost, red_x_test)

# test accuracy
red_acc <- mean(red_pred == red_y_test-3)
print(red_acc)

# test MAE
mmetric(red_pred, red_y_test-3, metric = "MAE")

#importance graph
red_importance_matrix <- xgb.importance(model = red_boost)
red_imp  = red_importance_matrix$Gain
red_bp = barplot(red_imp[11:1], main="red Wine", horiz=TRUE, col = 'white')
text(x=red_imp[11:1], y = red_bp, labels=colnames(red_wines)[as.integer(importance_matrix$Feature)+1][11:1], cex = 0.6, xpd = TRUE, pos = 2)

confusion_white_1 <- confusion(red_pred + 3,  red_y_test)
rownames(confusion_white_1) <- c(min(unique(red_wines$quality)):max(unique(red_wines$quality)), 'Precision')
colnames(confusion_white_1) <- c(min(unique(red_pred + 3)):max(unique(red_pred + 3)))
print(as.table(confusion_white_1))



################## white ##################
white_boost <- xgboost(data = white_x_train, label = white_y_train-3, max.depth = 2, eta = .05,
                     nthread = 2, nround = 50000, objective = "multi:softmax", num_class = 6)

white_pred <- predict(white_boost, white_x_test)

# validation predict
white_val_pred <- predict(white_boost, white_x_val)

# validation accuracy
white_val_acc <- mean(round(white_val_pred) == white_y_val-3)
print(white_val_acc)


# test accuracy
white_acc <- mean(round(white_pred) == white_y_test-3)
print(white_acc)

# test MAE
mmetric(white_pred, white_y_test-3, metric = "MAE")

#importance graph
white_importance_matrix <- xgb.importance(model = white_boost)
white_imp  = white_importance_matrix$Gain
white_bp = barplot(white_imp[11:1], main="white Wine", horiz=TRUE, col = 'white')
text(x=white_imp[11:1], y = white_bp, labels=colnames(white_wines)[as.integer(importance_matrix$Feature)+1][11:1], cex = 0.6, xpd = TRUE, pos = 2)

confusion_white_1 <- confusion(white_pred + 3,  white_y_test)
rownames(confusion_white_1) <- c(min(unique(white_wines$quality)):max(unique(white_wines$quality)), 'Precision')
colnames(confusion_white_1) <- c(min(unique(white_pred + 3)):max(unique(white_pred + 3)))
print(as.table(confusion_white_1))

