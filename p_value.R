# import data set
red_wines = read.csv("red_wines.csv", sep = ";")
white_wines = read.csv("white_wines.csv", sep = ";")

# standardize data to a zero mean and one standard deviation
red_wines[,(1:11)] <- scale(red_wines[,(1:11)])
white_wines[,(1:11)] <- scale(white_wines[,(1:11)])

################ univariate linear regression p-value ##################


red_p_value <- vector("numeric", 11)
for (i in 1:11){
  model <- lm(red_wines$quality ~ red_wines[,i])
  red_p_value[i] <- data.frame(summary(model)$coefficients)[2,4]
}

colnames(red_wines)[order(red_p_value)]

white_p_value <- vector("numeric", 11)
for (i in 1:11){
  model <- lm(white_wines$quality ~ white_wines[,i])
  white_p_value[i] <- data.frame(summary(model)$coefficients)[2,4]
}

colnames(white_wines)[order(white_p_value)]

