##### Package Installation 

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.
us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cra
n.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.
us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cra
n.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.
us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cra
n.r-project.org")

##### Loading Libraries 

library(tidyverse)
library(caret)
library(gridExtra)
library(lubridate)
library(knitr)
library(e1071)

##### Downloading the MovieLens Data

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),                        col.names = c("userId", "movieId", "rating", "timestamp"))

##### Creating the Dataset

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

##### Creating the Validation Set: 10% of the MovieLens Data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

##### Dataset Exploration: Inspecting the Dimensions of EDX 

head(edx)

cat("The edx dataset has", nrow(edx), "rows and", ncol(edx), "columns.\n")
cat("There are", n_distinct(edx$userId), "different users and", n_distinct(edx$movieId), "different movies in the edx dataset.")

# Check if edx has missing values

any(is.na(edx))

# What are the ratings year by year?

edx_year_rating <- edx %>% 
  transform (date = as.Date(as.POSIXlt(timestamp, origin = "1970-01-01", format = "%Y-%m-%d"), format = "%Y-%m-%d")) %>%
  mutate (year_month = format(as.Date(date), "%Y-%m"))

ggplot(edx_year_rating) + 
  geom_point(aes(x = date, y = rating)) +
  scale_x_date(date_labels = "%Y", date_breaks  = "1 year") +
  labs(title = "Ratings Year by Year", x = "Year", y = "Rating")

# What are the rating averages and medians year by year?

edx_yearmonth_rating <- edx_year_rating %>%
  group_by(year_month) %>%
  summarize(avg = mean(rating), median = median(rating))

ggplot(edx_yearmonth_rating) + 
  geom_point(aes(x = year_month, y = avg, colour = "avg")) +
  geom_point(aes(x = year_month, y = median, colour = "median")) +
  ylim(0, 5) +
  scale_x_discrete(breaks = c("1996-01", "1997-01", "1997-01", "1998-01", "1999-01", 
                              "2000-01", "2001-01", "2002-01", "2003-01", "2004-01", 
                              "2005-01", "2006-01", "2007-01", "2008-01", "2009-01")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0)) +
  labs(title = "Rating Averages and Medians, aggregated by month ", x = "Year", y = "Rating") 

##### Dataset Exploration: Ratings

summary(edx$rating)

# Distribution of ratings
ggplot(data = edx, aes(x = rating)) +
  geom_bar() + 
  labs(title = "Distribution of Ratings", x = "Rating", y = "Number of ratings")

##### Dataset Exploration: Movies 

edx_movies <- edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

summary(edx_movies$count)

##### Dataset Exploration: Users 

edx_users <- edx %>%
  group_by(userId) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

summary(edx_users$count)

##### Creating the Predictive Model

set.seed(699)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
# Use semi_join() to ensure that all users and movies in the test set are also in the training set
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

##### Mean Method

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

mu <- mean(train_set$rating)

cat("The average rating of all movies across all users is:", mu)

train_set %>% group_by(movieId) %>% filter(n()>=1000) %>% summarize(avg_rating = mean(rating)) %>% qplot(avg_rating, geom = "histogram", color = I("red"), bins=30, data = .)

predictions <- rep(mu, nrow(test_set))
naive_rmse <- RMSE(test_set$rating, predictions)
cat("The RMSE with mean method is:", naive_rmse)

##### Movie Effect Method

rmse_results <- data_frame(method = "Mean", RMSE = naive_rmse)

movie_means <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu)) 

qplot(b_i, geom = "histogram", color = I("black"), bins=25, data = movie_means)

joined <- test_set %>% left_join(movie_means, by='movieId')

predicted_ratings <- mu + joined$b_i

model1_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie effect model", RMSE = model1_rmse ))

rmse_results %>% kable

##### Movie Effect - Regularized

lambdas <- seq(0, 8, 0.25)

tmp <- train_set %>%
  group_by(movieId) %>%
  summarize(sum = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  joined <- test_set %>%
    left_join(tmp, by='movieId') %>%
    mutate(b_i = sum/(n_i+l))
  predicted_ratings <- mu + joined$b_i
  return(RMSE(predicted_ratings, test_set$rating))
})

cat("The best lambda (which minimizes the RMSE) for the movie effec
t is:", lambdas[which.min(rmses)])
qplot(lambdas, rmses)

lambda <- 2.75
movie_reg_means <- train_set %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

joined <- test_set %>% left_join(movie_reg_means, by='movieId') %>% replace_na(list(b_i=0))

predicted_ratings <- mu + joined$b_i

model1_reg_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect - Regularized", RMSE = model1_reg_rmse )) 

rmse_results %>% kable

##### Movie and User Effects - Regularized

lambdas_2 <- seq(0, 10, 0.25)

rmses <- sapply(lambdas_2, function(l){mu <- mean(train_set$rating)

b_i <- train_set %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+l)) 

b_u <- train_set %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- test_set %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% .$pred 
return(RMSE(predicted_ratings, test_set$rating)) 
})

cat("The best lambda_2 (which minimizes the RMSE) for the user and
movie effects is", lambdas_2[which.min(rmses)])

qplot(lambdas_2, rmses)

lambda_2 <- 5
user_reg_means <- train_set %>% left_join(movie_reg_means) %>% mutate(resids = rating - mu - b_i) %>% group_by(userId) %>% summarize(b_u = sum(resids)/(n()+lambda_2))

joined <- test_set %>% left_join(movie_reg_means, by='movieId') %>% left_join(user_reg_means, by='userId') %>% replace_na(list(b_i=0, b_u=0))

predicted_ratings <- mu + joined$b_i + joined$b_u

model2_reg_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results, data_frame(method = "Regularized movie and user effects model, lambda2 = 5", RMSE = model2_reg_rmse ))

rmse_results %>% kable