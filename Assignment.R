rm(list=ls())
dev.off()

### Libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(glmnet)
library(caretEnsemble)
library(pander)


### Setup

# paths
inp_path <- dirname(rstudioapi::getSourceEditorContext()$path)


# read data
train <- read.csv(paste(inp_path, "pml-training.csv", sep ="/"))
test <- read.csv(paste(inp_path, "pml-testing.csv", sep ="/"))

# quick exploration
dim(tain)
dim(test)

# options
#opt <- "exclude_none"
opt <- "exclude_cols"

### Subset data -> data which are mostly NA in train and always NA in test
# Check if any variable of type logical in train -> no
for (i in 1:ncol(train)){
  if(sapply(train, class)[i]=="logical"){
    print(names(train)[i])
  }
}
# extracting vector of names of variables of type logical in test (=variables which are all NA)
exclude <- c()
for (i in 1:ncol(test)){
  if(sapply(test, class)[i]=="logical"){
    exclude <- c(exclude, names(test)[i])
  }
}
# add redundant variables to vector
exclude <- c(exclude, names(test)[1])
# excluding variables of type logical from train and test
if(opt=="exclude_cols"){
  train <- train[,!(colnames(train) %in% exclude)]
  test <- test[,!(colnames(test) %in% exclude)]
}


### Re-Split Data into validation and training set

# reduce data
# 1 user
# re-code dependent variable: A against all
# train <- train %>%
#   filter(user_name == "adelmo") %>%
#   mutate(classe = as.factor(ifelse(classe=="A", "correct", "incorrect")))
# train <- train %>%
#   filter(user_name == "adelmo")

### build models
set.seed(98332)
# # reduce data: COMMENT OUT LATER
# in_reduced <- createDataPartition(y=train$classe,
#                                   p=0.1, list=F)
# reduced <- train[in_reduced,]
# train <- reduced
# re-split training-set into train & validation
train_tmp <- train
inTrain <- createDataPartition(y=train_tmp$classe,
                               p=0.7, list=F)
train <- train_tmp[inTrain,]
valid <- train_tmp[-inTrain,]
rm(train_tmp)


### Exploring data

dim(train)
table(train$user_name) # 6 subjects
table(train$classe) # balanced data set in terms of dependent variable
table(train$user_name, train$classe)

# Examining Clusters
# plotting variables explaining some variation in classe
numericClasse <- as.numeric(as.factor(train$classe))
vars <- c(14-1,18-1,19-1,20-1,24-1,42-1,44-1,46-1,48-1)
#vars <- c(9,18,19,20,24,26,30,42,48)
par(mfrow=c(3,3))
for(i in vars){
  (plot(train[,i],pch=19,col=numericClasse,ylab=names(train)[i]))
}

# SVD analysis
svd1 = svd(scale(train[,-c(2-1, 5-1, 6-1, 60-1)]))
par(mfrow=c(1,2))
plot(svd1$u[,1],col=numericClasse,pch=19) 
plot(svd1$u[,2],col=numericClasse,pch=19) 

# finding maximum contributor
par(mfrow=c(1,1))
plot(svd1$v[,2],pch=19)
maxContrib <- which.max(svd1$v[,2]) # column 296 is the maximum contributor (v=right singular vector assiciated with columns)
names(train[,-c(2, 5, 6, 60)])[maxContrib] # checking which variable is the maximum contributor


### Build Models


## Penalized Multinomial Regression
# tweak trainControl parameter to use cross validation
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                     savePredictions = TRUE, 
                     classProbs=TRUE)
# build model with cross-validation
mod_glm <- train(classe~.,  data=train, method="multinom",
                 trControl = ctrl, tuneLength = 5, trace = FALSE)
# predict on validation set
predictions_glm <- predict(mod_glm, newdata=valid)
stat_glm <- confusionMatrix(data=predictions_glm, valid$classe)
stat_glm
# save model
saveRDS(mod_glm, paste(inp_path, "mod_glm.rds", sep ="/"))

## svm
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                     savePredictions = TRUE, 
                     classProbs=TRUE)
mod_svm <- train(classe~., data=train, method = "svmLinear", trControl = ctrl, trace = FALSE)
head(mod_svm$pred)
predictions_svm <- predict(mod_svm, newdata=valid)
stat_svm <- confusionMatrix(data=predictions_svm, valid$classe)
stat_svm
# save model
saveRDS(mod_svm, paste(inp_path, "mod_svm.rds", sep ="/"))

#
## rf
# https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr:
# The out-of-bag (oob) error estimate
# In random forests, there is no need for cross-validation 
# or a separate test set to get an unbiased estimate of the 
# test set error. It is estimated internally, during the run, 
# as follows:
# Each tree is constructed using a different bootstrap sample 
# from the original data. About one-third of the cases are left 
# out of the bootstrap sample and not used in the construction 
# of the kth tree.
# Put each case left out in the construction of the kth tree 
# down the kth tree to get a classification. In this way, a 
# test set classification is obtained for each case in about 
# one-third of the trees. At the end of the run, take j to be 
# the class that got most of the votes every time case n was 
# oob. The proportion of times that j is not equal to the true 
# class of n averaged over all cases is the oob error estimate. 
# This has proven to be unbiased in many tests. 
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                        savePredictions = TRUE, 
                        classProbs=TRUE)
metric <- "Accuracy"
mtry <- sqrt(ncol(train))
mtry <- 3
tunegrid <- expand.grid(.mtry=mtry)

mod_rf <- train(classe~., data = train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, trace = FALSE)
print(mod_rf)
predictions_rf <- predict(mod_rf, valid) # storing predictions no test set necessary -> unbiased predictors due to inherent cross-validation in rf
stat_rf <- confusionMatrix(predictions_rf, as.factor(valid$classe))
stat_rf
# save model
saveRDS(mod_rf, paste(inp_path, "mod_rf.rds", sep ="/"))

# ## lasso
# # Additionnal data preparation
# # Dumy code categorical predictor variables
# x <- model.matrix(classe~., train)[,-1]
# xv <- model.matrix(classe~., valid)[,-1]
# # Convert the outcome (class) to a numerical variable
# y <- as.numeric(train$classe)
# yv <- as.numeric(valid$classe)
# #
# #
# # Find the best lambda using cross-validation
# cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "multinomial", type.multinomial = "grouped", parallel = TRUE)
# # Fit the final model on the training data
# mod_lasso <- glmnet(x, y, alpha = 1, family = "multinomial", type.multinomial = "grouped",
#                     lambda = cv.lasso$lambda.min)
# #coef(mod_lasso)
# predictions_lasso <- predict(mod_lasso, xv, type = "class")
# stat_lasso <- confusionMatrix(as.factor(as.numeric(predictions_lasso)), as.factor(yv))
# stat_lasso
# # save model
# save(mod_lasso, file = paste(inp_path, "mod_lasso.RData", sep ="/"))


# ## Checking models together (on train-set only)
# trainControl <- trainControl(method="repeatedcv", 
#                              number=10, 
#                              repeats=3,
#                              savePredictions=TRUE, 
#                              classProbs=TRUE)
# algorithmList <- c('rf', 'svmLinear', 'multinom')
# mod_stacked <- caretList(classe~., data = train, trControl=trainControl, methodList=algorithmList) 
# results <- resamples(mod_stacked)
# summary(results)
# scales <- list(x=list(relation="free"), y=list(relation="free"))
# bwplot(results, scales=scales)


## plot results
# confusion matrices
pandoc.table(stat_glm$table, style="rmarkdown")
pandoc.table(stat_svm$table, style="rmarkdown")
pandoc.table(stat_rf$table, style="rmarkdown")
# comparing accuracy & kappa tables
stats <- as.data.frame(rbind(stat_glm$overall, stat_svm$overall, stat_rf$overall))
stats$model <- ""
stats$model[1] <- "glm"
stats$model[2] <- "svm"
stats$model[3] <- "rf"
stats <- stats %>%
  select(model, Accuracy, Kappa)
pandoc.table(stats, style="rmarkdown")
# variable importance of best model (random forest)
var_imp <- varImp(mod_rf, scale = T)
vars <- row.names(var_imp$importance)[var_imp$importance$Overall %in% sort(var_imp$importance$Overall, 
                                                   decreasing = T)[1:9]]
# Examining Clusters
# plotting variables explaining some variation in classe
numericClasse <- as.numeric(as.factor(train$classe))
#vars <- c(9,18,19,20,24,26,30,42,48)
par(mfrow=c(3,3))
for(i in vars){
  print(plot(train[,i],pch=19,col=numericClasse,ylab=i))
}

