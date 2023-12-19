library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(GGally)
library(corrplot)
library(patchwork)

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


train %>%
  group_by(Soil_Type) %>%
  summarize(count = n()) %>%
  filter(count < 50) %>%
  select(Soil_Type) %>%
  c() -> levels



train %>%
  select(-c(Cover_Type, Soil_Type, Wilderness_Area1,
            Wilderness_Area2, Wilderness_Area3, Wilderness_Area4)) %>%
  rename(hor_dist_hydro = Horizontal_Distance_To_Hydrology,
         ver_dist_hydro = Vertical_Distance_To_Hydrology,
         hor_dist_road = Horizontal_Distance_To_Roadways,
         hor_dist_fire = Horizontal_Distance_To_Fire_Points) %>%
  cor() %>%
  corrplot(method = "number")


train %>%
  ggplot(aes(x = Cover_Type,
             y = Vertical_Distance_To_Hydrology)) +
  geom_boxplot(fill = "lightblue") -> plot1
train %>%
  ggplot(aes(x = Cover_Type,
             y = Horizontal_Distance_To_Hydrology)) +
  geom_boxplot(fill = "orange") -> plot2
train %>%
  ggplot() +
  geom_point(aes(x = Vertical_Distance_To_Hydrology,
                 y = Horizontal_Distance_To_Hydrology)) -> plot3
(plot1 | plot2) / plot3
# create new variable "distance" that uses pathagorean theorem 
# to find the distance to hydrology and then create a variable that indicates if 
# the nearest source of water is up or down

train %>%
  mutate(distance = sqrt(Vertical_Distance_To_Hydrology^2 +
                           Horizontal_Distance_To_Hydrology^2)) %>%
  ggplot(aes(x = Cover_Type, y = distance)) +
  geom_boxplot(fill = "lightblue") -> plot4


train %>%
  ggplot() +
  geom_point(aes(x = Hillshade_9am, y = Hillshade_3pm)) -> plot3
train %>%
  select(Hillshade_9am, Hillshade_3pm, Cover_Type) %>%
  ggplot(mapping = aes(x = Cover_Type, y = Hillshade_9am)) +
  geom_boxplot(fill = "lightblue") -> plot1
train %>%
  select(Hillshade_9am, Hillshade_3pm, Cover_Type) %>%
  ggplot(mapping = aes(x = Cover_Type, y = Hillshade_3pm)) +
  geom_boxplot(fill = "orange") -> plot2

(plot1 | plot2) / plot3

# Because Hillshade_3pm has greater variances we are going to throw it out
# in favor of Hillshade_9pm



train %>%
  group_by(Soil_Type) %>%
  summarize(count = n()) %>%
  filter(count < 100)












# ----- target encoding
train <- vroom('train.csv')
test <- vroom('test.csv')

# Remove the dummy encoding for Soil_Type
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


# Create a recipe
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

tunedModel <- control_stack_resamples() 
untuned_model <- control_stack_grid()

folds <- vfold_cv(train, v = 5)

# Penalized Regression model

reg_mod <- multinom_reg(penalty = 0.0000000001,
                        mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

reg_results <- workflow() %>%
  add_recipe(fc_recipe) %>%
  add_model(reg_mod) %>%
  fit(train)

preds <- predict(reg_results, new_data = test)

test %>%
  mutate(Cover_Type = preds$.pred_class) %>%
  select(Id, Cover_Type) %>%
  vroom_write("submission.csv", delim = ",")


