library(utf8)
library(dplyr)
library(quanteda)
library(quanteda.textmodels)
library(caret)
library(tidyr)
library(tidytext)


# Load the data
drug_train <- read.csv("UCIdrug_train.csv")
drug_test <- read.csv("UCIdrug_test.csv")
drug <- read.csv("UCIdrugs.csv")

head(drug)

# Column selection
reviews_train <- pull(select(drug_train, "review"))
reviews_test <- pull(select(drug_test, "review"))

# Check the encoding of the reviews for training and test texts
reviews_train[!utf8_valid(reviews_train)]
reviews_test[!utf8_valid(reviews_test)]

# Check character normalization (Normalized Composed Form) for training and test texts
reviews_train_NFC <- utf8_normalize(reviews_train)
reviews_test_NFC <- utf8_normalize(reviews_test)

sum(reviews_train_NFC != reviews_train)
sum(reviews_test_NFC != reviews_test)

# Use data as corpus
corpus <- corpus(drug, text_field = "review")

# Transform corpus into DFM
dfmat <- dfm(tokens(corpus) %>% tokens_tolower(),
             remove_punct = TRUE, remove_numbers = TRUE,  remove_symbols = TRUE) %>%
  dfm_remove(stopwords('english'))

# Prior to the classification, we can do some sentiment analysis
dfmat_sentiments <- tidy(dfmat)

# Count of different positive and negative words for each text
dfmat_sentiments <- dfmat_sentiments %>%
  inner_join(get_sentiments("bing"), by = c(term = "word"))

dfmat_sentiments

# Detect the documents with the most negative feelings
dfmat_sentiments %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(sentiment)

# Plot the contribution of the words to a positive or negative feeling
dfmat_sentiments %>%
  count(sentiment, term, wt = count) %>%
  filter(n >= 7000) %>%
  mutate(n = ifelse(sentiment == "negative", -n, n)) %>%
  mutate(term = reorder(term, n)) %>%
  ggplot(aes(term, n, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylab("Contribution to sentiment")



# Split into train and test to train the machine learning models
dfm_train <- dfm_subset(dfmat, set == "train")
dfm_test <- dfm_subset(dfmat, set == "test")


# Naive Bayes
model_nb <- textmodel_nb(dfm_train,
                      dfm_train$label,
                      distribution = "Bernoulli")
pred_nb <- predict(model_nb,
                newdata = dfm_test)

confM_nb <- confusionMatrix(table(pred_nb, docvars(dfm_test)$label))

acc_coincidences <- sum(as.character(pred_nb) == as.character(docvars(dfm_test)$label))

acc_total <- length(as.character(pred_nb))
acc_nb <- acc_coincidences/acc_total
acc_nb

confM_nb[["byClass"]]




# Support Vector Machine (SVM)
set.seed(23)
svmPredictions <- function(x,
                           weight){ #weight can be "uniform", "docfreq" or "termfreq".
  
  # Sample of documents 
  dfmat_train <- dfm_sample(dfm_subset(dfmat, set == "train"), x)
  
  dfmat_test <- dfm_subset(dfmat, set == "test")
  
  # Train the SVM model with the sample
  model_svm <- textmodel_svm(dfmat_train,
                             dfmat_train$label,
                             weight = weight)
  
  # Prediction of the labels
  pred_svm <- predict(model_svm,
                      newdata = dfmat_test)
  
  # Compute the confusion matrix 
  confM_svm <- confusionMatrix(table(pred_svm, docvars(dfmat_test)$label))
  
  # Compute the accuracy
  acc_coincidences <- sum(as.character(pred_svm) == as.character(docvars(dfmat_test)$label))
  acc_total <- length(as.character(pred_svm))
  acc_svm <- acc_coincidences/acc_total
  acc_svm
  
  # Show the metrics by class
  confM_svm[["byClass"]]
  
}

# Results for a sample size of 10000 and uniform weight
svmPredictions(10000, "uniform")

# Lists to save the precision and recall values for different sample sizes
results_recall_bad <- list()
results_recall_neutral <- list()
results_recall_good <- list()

results_precision_bad <- list()
results_precision_neutral <- list()
results_precision_good <- list()


for (i in seq(100, 10000, by = 100)){
  set.seed(23)
  
  # Sample of documents 
  dfmat_train <- dfm_sample(dfm_subset(dfmat, set == "train"), i)
  dfmat_test <- dfm_subset(dfmat, set == "test")
  
  # Train the SVM model with the sample
  model_svm <- textmodel_svm(dfmat_train,
                         dfmat_train$label,
                         weight = "uniform")
  
  # Prediction of the labels
  pred_svm <- predict(model_svm,
                  newdata = dfmat_test)
  
  # Compute the confusion matrix 
  confM_svm <- confusionMatrix(table(pred_svm, docvars(dfmat_test)$label))
  
  # Store the metrics on the lists
  results_recall_bad <- append(results_recall_bad, confM_svm$byClass[1, 1])
  results_recall_neutral <- append(results_recall_neutral, confM_svm$byClass[3, 1])
  results_recall_good <- append(results_recall_good, confM_svm$byClass[2, 1])
  
  results_precision_bad <- append(results_precision_bad, confM_svm$byClass[1, 3])
  results_precision_neutral <- append(results_precision_neutral, confM_svm$byClass[3, 3])
  results_precision_good <- append(results_precision_good, confM_svm$byClass[2, 3])
}


# Store data in a data frame
df <-data.frame(sample_size = seq(100, 10000, by = 100))
df$precision_bad = unlist(results_precision_bad)
df$precision_neutral = unlist(results_precision_neutral)
df$precision_good = unlist(results_precision_good)
df$recall_bad = unlist(results_recall_bad)
df$recall_neutral = unlist(results_recall_neutral)
df$recall_good = unlist(results_recall_good)

# Plot the precision evolution
ggplot(df, aes(x = sample_size)) + 
  geom_line(data = df, aes(y = precision_bad, colour = "Bad")) +
  geom_line(data = df, aes(y = precision_neutral, colour = "Neutral")) +
  geom_line(data = df, aes(y = precision_good, colour = "Good")) +
  xlab("Sample size") +
  ylab("Precision for each class") +
  theme_minimal()+
  ggtitle("Sample size vs precision for each class") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_colour_manual("", 
                      breaks = c("Bad", "Neutral",  "Good"),
                      values = c("orange", "gray", "blue")) 
  
# Plot the recall evolution
ggplot(df, aes(x = sample_size)) + 
  geom_line(data = df, aes(y = recall_bad, colour = "Bad")) +
  geom_line(data = df, aes(y = recall_neutral, colour = "Neutral")) +
  geom_line(data = df, aes(y = recall_good, colour = "Good")) +
  xlab("Sample size") +
  ylab("Recall for each class") +
  theme_minimal()+
  ggtitle("Sample size vs recall for each class") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_colour_manual("", 
                      breaks = c("Bad", "Neutral",  "Good"),
                      values = c("orange", "gray", "blue")) 
