library(tidyverse)
library(vroom)
library(tidymodels)
library(doParallel)
library(embed)

cl <- makePSOCKcluster(10)
registerDoParallel(cl)

train <- vroom('train.csv')
test <- vroom('test.csv')

un_dummy <- function(df, names){
  num = 1
  for(i in names){
    if(num == 1){
      df %>%
        mutate(Soil_Type = if_else(!!sym(i) == 1, num, 0)) %>%
        select(-c(!!sym(i)))-> df
    }
    else{
      df %>%
        mutate(Soil_Type = if_else(!!sym(i) == 1, num, Soil_Type)) %>%
        select(-c(!!sym(i))) -> df
    }
    num = num + 1
  }
  df %>%
    mutate(Soil_Type = as_factor(Soil_Type)) -> df
  return(df) 
}

names = rep('0', 40)
for(i in 1:40){
  names[i] = paste("Soil_Type", i, sep = "")
}

train <- un_dummy(train, names) %>%
  mutate(Cover_Type = as_factor(Cover_Type))
test <- un_dummy(test, names)

fc_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_mutate(distance = sqrt(Vertical_Distance_To_Hydrology^2 +
                                Horizontal_Distance_To_Hydrology^2)) %>%
  step_mutate(above_water = if_else(Vertical_Distance_To_Hydrology > 0,
                                    1, 0)) %>%
  step_rm(c(Vertical_Distance_To_Hydrology ,
            Horizontal_Distance_To_Hydrology,
            Hillshade_3pm)) %>%
  step_nzv(all_predictors()) %>% 
  step_mutate()
  step_lencode_glm(Soil_Type, outcome = vars(Cover_Type)) %>%
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
