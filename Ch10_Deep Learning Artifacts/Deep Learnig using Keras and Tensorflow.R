## ----setup, eval = FALSE-------------------------------------------------
## 
## #Installing Keras
## 
## install.packages("keras")
## library(keras)
## install_keras()
## 
## #Install TensorFlow as a Backend for Keras. By default it takes CPU but you could also mention, "tensorflow = "gpu"
## 
## install_keras(tensorflow)
## 

## ----preprocessing_1-----------------------------------------------------

#loading keras
library(keras)

#Data can be downloaded from
#https://www.kaggle.com/c/quora-question-pairs/data

setwd("C:\\Users\\Karthik\\OneDrive - Data Science and Analytics Consulting LLP\\Book - Machine Learning Using R - 2nd Edition\\Deep Learning using Keras and Tensorflow\\Data\\quora_questions")

quora_data <- read.csv("train.csv")

quora_data$question1 = as.character(quora_data$question1)
quora_data$question2 = as.character(quora_data$question2)

# Example Question Pairs
quora_data$question1[1]
quora_data$question2[1]

#Keras tokenizer

tokenizer <- text_tokenizer(num_words = 50000)
tokenizer %>% fit_text_tokenizer(unique(c(quora_data$question1, quora_data$question2)))

#Text tokenizer to transform each question1 and question2 into a list of integers

question1 <- texts_to_sequences(tokenizer, quora_data$question1)
question2 <- texts_to_sequences(tokenizer, quora_data$question2)


## ----preprocessing_2, warning=FALSE--------------------------------------

library(purrr)
questions_length <- c(
  map_int(question1, length),
  map_int(question2, length)
)

## ----preprocessing_3-----------------------------------------------------

#80th, 90th, 95th and 99th Quantiles

quantile(questions_length, c(0.8, 0.9, 0.95, 0.99))

# Padding Lenght = 20
question1_padded <- pad_sequences(question1, maxlen = 20, value = 50000 + 1)
question2_padded <- pad_sequences(question2, maxlen = 20, value = 50000 + 1)

## ----logit---------------------------------------------------------------

percentage_words_question1 <- map2_dbl(question1, question2, ~mean(.x %in% .y))
percentage_words_question2 <- map2_dbl(question2, question1, ~mean(.x %in% .y))

quora_logit_model <- data.frame(
  percentage_words_question1 = percentage_words_question1,
  percentage_words_question2 = percentage_words_question2,
  is_duplicate = quora_data$is_duplicate
) %>%
  na.omit()

#With this input variables, we'll create the logistic model. We will take 10% of the data as a sample for validation.

val_sample <- sample.int(nrow(quora_logit_model), 0.1*nrow(quora_logit_model))
quora_logistic_regression <- glm(
  is_duplicate ~ percentage_words_question1 + percentage_words_question2, 
  family = "binomial",
  data = quora_logit_model[-val_sample,]
)
summary(quora_logistic_regression)

#Calculate the accuracy on our validation set

pred <- predict(quora_logistic_regression, quora_logit_model[val_sample,], type = "response")
pred <- pred > mean(quora_logit_model$is_duplicate[-val_sample])
accuracy <- table(pred, quora_logit_model$is_duplicate[val_sample]) %>% 
  prop.table() %>% 
  diag() %>% 
  sum()
accuracy


## ----siemese_archi, warning=FALSE----------------------------------------
# Inputs of the model

input1 <- layer_input(shape = c(20), name = "input_question1")
input2 <- layer_input(shape = c(20), name = "input_question2")

#Embed the questions in a vector

word_embedder <- layer_embedding( 
  input_dim = 50000 + 2, # vocab size + UNK token + padding value
  output_dim = 128,      # hyperparameter - embedding size
  input_length = 20,     # padding size,
  embeddings_regularizer = regularizer_l2(0.0001) # hyperparameter - regularization 
)

#LSTM Layer

seq_embedder <- layer_lstm(
  units = 128, # hyperparameter -- sequence embedding size
  kernel_regularizer = regularizer_l2(0.0001) # hyperparameter - regularization 
)

#Define the relationship between the input vectors and the embeddings layers. #Here we use the same layers and weights on both inputs - Siamese network. 
#Even if we switch question1 and question2, the architecture makes sure that we don't get two different outputs

vector1 <- input1 %>% word_embedder() %>% seq_embedder()
vector2 <- input2 %>% word_embedder() %>% seq_embedder()

#Cosine similarity is used. The syntax shows that its a dot product of the two vectors without the normalization part.

cosine_similarity <- layer_dot(list(vector1, vector2), axes = 1)

# Final sigmoid layer to output the probability of both questions

output <- cosine_similarity %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Keras model defined in terms of it's inputs and outputs and compile it.
# Minimize the logloss equivalent to minimizing the binary crossentropy 
# Using the Adam optimizer

model <- keras_model(list(input1, input2), output)
model %>% compile(
  optimizer = "adam", 
  metrics = list(acc = metric_binary_accuracy), 
  loss = "binary_crossentropy"
)

# Model Summary

summary(model)


#Sample for validation before model fitting

set.seed(1817328)
val_sample <- sample.int(nrow(question1_padded), size = 0.1*nrow(question1_padded))

train_question1_padded <- question1_padded[-val_sample,]
train_question2_padded <- question2_padded[-val_sample,]
train_for_duplicate <- quora_data$is_duplicate[-val_sample]

val_question1_padded <- question1_padded[val_sample,]
val_question2_padded <- question2_padded[val_sample,]
val_is_duplicate <- quora_data$is_duplicate[val_sample]

# fit() function to train the model:

model %>% fit(
  list(train_question1_padded, train_question2_padded),
  train_for_duplicate, 
  batch_size = 64, 
  epochs = 10, 
  validation_data = list(
    list(val_question1_padded, val_question2_padded), 
    val_is_duplicate
  )
)

#We could save our model for inference with the save_model_hdf5() function.

save_model_hdf5(model, "model-question-pairs.hdf5")


## ----predictions_1-------------------------------------------------------

library(keras)
model <- load_model_hdf5("model-question-pairs.hdf5", compile = FALSE)
tokenizer <- load_text_tokenizer("tokenizer-question-pairs")

## ----predictions_2-------------------------------------------------------
predict_question_pairs <- function(model, tokenizer, question1, question2) {
  question1 <- texts_to_sequences(tokenizer, list(question1))
  question2 <- texts_to_sequences(tokenizer, list(question2))
  
  question1 <- pad_sequences(question1, 20)
  question2 <- pad_sequences(question2, 20)
  
  as.numeric(predict(model, list(question1, question2)))
}

#Example 1
predict_question_pairs(
  model,
  tokenizer,
  "What is Machine Learning?",
  "What is Deep Learning?"
)

#Example 2
predict_question_pairs(
  model,
  tokenizer,
  "What is Machine Learning",
  "What are Machine Learning algorithms"
)

#Example 3
predict_question_pairs(
  model,
  tokenizer,
  "What is a Machine Learning",
  "What is a Machine Learning algorithms"
)



