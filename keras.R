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

library(keras)

# Get the one-hot encode of y
red_y_train <- to_categorical(red_wines_train$quality-3, 6)
red_y_test <- to_categorical(red_wines_test$quality-3, 6)

white_y_train <- to_categorical(white_wines_train$quality-3, 7)
white_y_test <- to_categorical(white_wines_test$quality-3, 7)

# drop label for each data
red_x_train <- red_wines_train[,(1:11)]
red_x_test <- red_wines_test[,(1:11)]

white_x_train <- white_wines_train[,(1:11)]
white_x_test <- white_wines_test[,(1:11)]

red_x_train <- t(t(array(red_x_train)))
red_x_test <- t(t(array(red_x_test)))

white_x_train <- t(t(array(white_x_train)))
white_x_test <- t(t(array(white_x_test)))


red_model <- keras_model_sequential() 
red_model %>% 
  layer_dense(units = 24, activation = 'relu', input_shape = 11) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 12, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 6, activation = 'softmax')

summary(red_model)

red_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = "accuracy"
)

history <- red_model %>% fit(
  red_x_train, red_y_train, 
  epochs = 30, batch_size = 10,
  validation_split = 0.2
)

red_model %>% evaluate(red_x_test, red_y_test)

white_model <- keras_model_sequential() 
white_model %>% 
  layer_dense(units = 30, activation = 'relu', input_shape = 11) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 15, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 7, activation = 'softmax')

summary(white_model)

white_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = "accuracy"
)

history <- white_model %>% fit(
  white_x_train, white_y_train, 
  epochs = 60, batch_size = 20,
  validation_split = 0.2
)

white_model %>% evaluate(white_x_test, white_y_test)