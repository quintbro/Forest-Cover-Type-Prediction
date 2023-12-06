library(tidyverse)
library(vroom)
library(tidymodels)
library(doParallel)

cl <- makePSOCKcluster(10)
registerDoParallel(cl)

train <- vroom('train.csv')
test <- vroom('test.csv')

train %>%
  mutate(Cover_Type = as_factor(Cover_Type)) -> train

fc_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_mutate(distance = sqrt(Vertical_Distance_To_Hydrology^2 +
                                Horizontal_Distance_To_Hydrology^2)) %>%
  step_mutate(above_water = if_else(Vertical_Distance_To_Hydrology > 0,
                                    1, 0)) %>%
  step_rm(c(Vertical_Distance_To_Hydrology ,
            Horizontal_Distance_To_Hydrology,
            Hillshade_3pm)) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


fc_mod <- rand_forest(mtry = tune(),
                     min_n = tune(),
                     trees = 600) %>%
  set_engine("ranger") %>%
  set_mode("classification")

fc_wf <- workflow() %>%
  add_model(fc_mod) %>%
  add_recipe(fc_recipe)

tuning_grid <- grid_regular(mtry(range = range(1,54)),
                            min_n(),
                            levels = 1)

folds <- vfold_cv(train, v = 5)

cv_results <- fc_wf %>%
  tune_grid(grid = tuning_grid,
            resamples = folds,
            metrics = metric_set(accuracy))

params <- cv_results %>%
  select_best("accuracy")

final_wf <- fc_wf %>%
  finalize_workflow(params) %>%
  fit(train)

preds <- predict(final_wf, new_data = test)

test %>%
  mutate(Cover_Type = preds$.pred_class) %>%
  select(Id, Cover_Type) %>%
  vroom_write("submission.csv", delim = ",")

stopCluster(cl)
