# import data set
library(readr)
red_wines <- read_delim("red_wines.csv", ";", escape_double = FALSE, trim_ws = TRUE)
white_wines <- read_delim("white_wines.csv", ";", escape_double = FALSE, trim_ws = TRUE)

# standardize data to a zero mean and one standard deviation
reds <- scale(red_wines[,(1:11)])
reds <- as.data.frame(reds)
whites <-scale(white_wines[,(1:11)])
whites <- as.data.frame(whites)

# Multiple Linear Regression
reds_MR <- lm(formula = red_wines$quality ~ ., data = reds)
whites_MR <- lm(formula = white_wines$quality ~ ., data = whites)