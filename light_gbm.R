library(tidyverse)
library(vroom)
library(tidymodels)
library(lightgbm)
library(bonsai)
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
            Hillshade_3pm, Id)) %>%
  step_zv(all_predictors()) %>%
  step_novel(Soil_Type, new_level = "other")%>%
  step_other(Soil_Type, threshold = 50) %>%
  step_lencode_glm(Soil_Type, outcome = vars(Cover_Type)) %>%
  step_normalize(all_numeric_predictors())


fc_mod <- boost_tree(learn_rate = 0.22, # Current best: 0.22
                     tree_depth = 15,
                     trees = 600) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

fc_wf <- workflow() %>%
  add_model(fc_mod) %>%
  add_recipe(fc_recipe) %>%
  fit(train)

preds <- predict(fc_wf, new_data = test)

test %>%
  mutate(Cover_Type = preds$.pred_class) %>%
  select(Id, Cover_Type) %>%
  vroom_write("submission.csv", delim = ",")

stopCluster(cl)
