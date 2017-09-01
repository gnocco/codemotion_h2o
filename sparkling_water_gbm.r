# Demo for Codemotion 2017 Milan by Gabriele Nocco
# @gnocco

### Connection to Spark/H2O cluster ###

library(rsparkling)
library(h2o)
library(dplyr)
library(sparklyr)

# for Spark 2.0.x
options(rsparkling.sparklingwater.version = "2.0.2")

sc <- spark_connect(master = "local[*]")

### ETL - Preprocessing ###

# on Spark
iris_tbl <- copy_to(sc, iris, "iris", overwrite = TRUE)
iris_tbl

# on H2O
iris_hf <- as_h2o_frame(sc, iris_tbl, strict_version_check = FALSE)

y <- "Species"
x <- setdiff(names(iris_hf), y)
iris_hf[,y] <- as.factor(iris_hf[,y])

# default spit is 75/25%
splits <- h2o.splitFrame(iris_hf, seed = 1)

### Train the Model ###

gbm_model <- h2o.gbm(x = x, 
                     y = y,
                     training_frame = splits[[1]],
                     validation_frame = splits[[2]],                     
                     ntrees = 20,
                     max_depth = 3,
                     learn_rate = 0.01,
                     col_sample_rate = 0.7,
                     seed = 1)


h2o.confusionMatrix(gbm_model, valid = TRUE)


### Grid search ###

gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))

# Train and validate a grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid1",
                      training_frame = splits[[1]],
                      validation_frame = splits[[1]],
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by validation MSE
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1", 
                             sort_by = "mse", 
                             decreasing = FALSE)

print(gbm_gridperf1)


### Save a Model ###


h2o.saveModel(gbm_model, path = "/home/gnocco")

h2o.download_pojo(gbm_model, path = "/home/gnocco")


### Disconnect ###

spark_disconnect_all()

