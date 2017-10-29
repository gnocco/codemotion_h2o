if (! ("h2o" %in% rownames(installed.packages()))) { install.packages("h2o") }

# point to the prostate data set in the h2o folder - no need to load h2o in memory yet
prosPath = system.file("extdata", "prostate.csv", package = "h2o")
prostate_df <- read.csv(prosPath)

# We don't need the ID field
prostate_df <- prostate_df[,-1]
summary(prostate_df)

# Split in traing and test set
set.seed(1234)
random_splits <- runif(nrow(prostate_df))
train_df <- prostate_df[random_splits < .5,]
dim(train_df)

validate_df <- prostate_df[random_splits >=.5,]
dim(validate_df)

#Random Forest on variable 'CAPSULE'
if (! ("randomForest" %in% rownames(installed.packages()))) { install.packages("randomForest") }
library(randomForest)

outcome_name <- 'CAPSULE'
feature_names <- setdiff(names(prostate_df), outcome_name)
set.seed(1234)
rf_model <- randomForest(x=train_df[,feature_names],
                         y=as.factor(train_df[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")


# pROC to calcultate AUC

if (! ("pROC" %in% rownames(installed.packages()))) { install.packages("pROC") }
library(pROC)

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions[,2])

x11()
plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')


#H2O encoder

library(h2o)
localH2O = h2o.init()

prostate.hex<-as.h2o(train_df, destination_frame="train.hex")


##Deep-learning
prostate.dl = h2o.deeplearning(x = feature_names, training_frame = prostate.hex,
                               autoencoder = TRUE,
                               reproducible = T,
                               seed = 1234,
                               hidden = c(6,5,6), epochs = 50)


# h2o.anomaly

prostate.anon = h2o.anomaly(prostate.dl, prostate.hex, per_feature=FALSE)
head(prostate.anon)
err <- as.data.frame(prostate.anon)

x11()
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')


# rebuild train_df_auto with best observations
train_df_auto <- train_df[err$Reconstruction.MSE < 0.1,]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_known <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_known[,2])

x11()
plot(auc_rf, print.thres = "best", main=paste('AUC(1):',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')



# rebuild train_df_auto with best observations
train_df_auto <- train_df[err$Reconstruction.MSE >= 0.1,]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_unknown <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_unknown[,2])

x11()
plot(auc_rf, print.thres = "best", main=paste('AUC(2):',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

# bagging both prediction sets (adding both prediction vectors together and dividing the total by two)
valid_all <- (validate_predictions_known[,2] + validate_predictions_unknown[,2]) / 2

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=valid_all)

x11()
plot(auc_rf, print.thres = "best", main=paste('AUC(3):',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')