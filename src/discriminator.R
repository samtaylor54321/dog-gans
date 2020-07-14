library(keras)
library(magrittr)
library(R6)


Discriminator <- R6Class("Discriminator", 
  public = list(
    discriminator = NULL,
    
    initialize = function(height, width, channels) {
      private$height = height
      private$width = width
      private$channels = channels
      
      private$discriminator_input = 
        layer_input(shape = c(private$height, private$width, private$channels))
      
      private$discriminator_output = private$discriminator_input %>% 
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
      
      private$discriminator_optimiser <- optimizer_rmsprop(
        lr = 0.00008,
        clipvalue = 1.0,
        decay = 1e-8)
      
      self$discriminator = keras_model(private$discriminator_input,
                                       private$discriminator_output) %>% 
        compile(optimizer = private$discriminator_optimiser, 
                loss ="binary_crossentropy") %>% 
        freeze_weights()
    }
  ),
  
  private = list(
    width = NA, 
    height = NA, 
    channels = NA,
    discriminator_input = NA,
    discriminator_output = NA,
    discriminator_optimiser = NA
  )
)
