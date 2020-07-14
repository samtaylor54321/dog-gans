library(keras)
library(R6)
library(yaml)

Gan <- R6Class("Gan", 
  public = list(
    
    initialize = function(width, height, channels, latent_dim) {
      width = width
      height = height
      channels = channels 
      latent_dim = latent_dim
    },
    
    generator = keras_model(private$generator_input, private$generator_ouput)
  ),
  
  private = list(
    # Input for latent dim for the generator
    generator_input = layer_input(shape = c(public$latent_dim)),
    
    # Output for latent dim for the generator
    generator_output = generator_input %>% 
      
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
  )
)
