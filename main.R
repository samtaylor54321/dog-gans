library(here)
library(imager)
library(glue)
library(purrr)
library(yaml)
library(keras)

map(paste("src/",list.files(here("src")), sep=""), source)

# Load settings for script
config <- yaml.load_file(here("settings.yaml"))

latent_dim <- 32
height <- 32
width <- 32
channels <- 3


generator_input <- layer_input(shape = c(latent_dim))

generator_output <- generator_input %>% 
  
  layer_dense(units = 128 * 16 * 16) %>% 
  layer_activation_leaky_relu() %>% 
  layer_reshape(target_shape = c(16, 16, 128)) %>% 
  
  layer_conv_2d(filters = 256, kernel_size = 5,
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  
  layer_conv_2d_transpose(filters = 256, kernel_size = 4,
                          strides = 2, padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  
  layer_conv_2d(filters = 256, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 256, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  
  layer_conv_2d(filters = channels, kernel_size = 7, 
                activation = "tanh", padding = "same") 

generator <- keras_model(generator_input, generator_output)



discriminator_input <- layer_input(shape = c(height, width, channels))

discriminator_output <- discriminator_input %>% 
  
  layer_conv_2d(filters = 128, kernel_size = 3) %>% 
  layer_activation_leaky_relu() %>% 
  
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>% 
  layer_activation_leaky_relu() %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 1, activation = "sigmoid")

discriminator <- keras_model(discriminator_input, discriminator_output)

discriminator_optimiser <- optimizer_rmsprop(
  lr = 0.00008,
  clipvalue = 1.0,
  decay = 1e-8)

discriminator %>% 
  compile(optimizer = discriminator_optimiser, loss ="binary_crossentropy")


freeze_weights(discriminator)

gan_input <- layer_input(shape = c(latent_dim))
gan_output <- discriminator(generator(gan_input))
gan <- keras_model(gan_input, gan_output)

gan_optimiser <- optimizer_rmsprop(
  lr = 0.0004,
  clipvalue = 1.0,
  decay = 1e-8
)

gan %>% compile(optimizer = gan_optimiser,
                loss = "binary_crossentropy")


source(here("src/gan.R"))
