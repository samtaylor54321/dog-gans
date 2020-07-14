library(keras)
library(magrittr)
library(R6)


Generator <- R6Class("Generator", 
  public = list(
    generator = NA,
      
    initialize = function(latent_dim, channels) {
      private$latent_dim = latent_dim
      private$channels = channels
      private$generator_input = layer_input(shape = private$latent_dim)
      private$generator_output = private$generator_input %>% 
        
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
        
        layer_conv_2d(filters = private$channels, kernel_size = 7, 
                      activation = "tanh", padding = "same")
      
      self$generator = keras_model(private$generator_input,
                                   private$generator_output)
    }
  ),
  private = list(
    latent_dim = NA, 
    channels = NA, 
    generator_input = NA,
    generator_output = NA
  )
)   
      



