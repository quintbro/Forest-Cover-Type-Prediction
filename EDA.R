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

# target encode the variables
# start by creating dummies for all levels of the target
train %>%
  mutate(spruce = if_else(Cover_Type == 1, 1, 0),
         lodge_pine = if_else(Cover_Type == 2, 1, 0),
         pond_pine = if_else(Cover_Type == 3, 1, 0),
         cotton = if_else(Cover_Type == 4, 1, 0),
         aspen = if_else(Cover_Type == 5, 1, 0),
         doug = if_else(Cover_Type == 6, 1, 0),
         krum = if_else(Cover_Type == 7, 1, 0)) %>%
  select(c(spruce, lodge_pine, pond_pine,
           cotton, aspen, doug, krum, Soil_Type)) -> target 
