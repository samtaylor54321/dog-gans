library(keras)
library(R6)


Gan <- R6Class("Gan", 
  public = list(
    gan = NA,
    
    initialize = function(latent_dim, discriminator, generator) {
      private$latent_dim = latent_dim
      private$discriminator = discriminator
      private$generator = generator
      
      private$gan_input = layer_input(shape = c(private$latent_dim))
      
      private$gan_output = private$discriminator(private$generator(
        private$gan_input))
      
      private$optimiser = optimizer_rmsprop(
        lr = 0.0004,
        clipvalue = 1.0,
        decay = 1e-8
      )
      
      self$gan = keras_model(private$gan_input, private$gan_output) %>% 
        compile(optimizer = private$optimiser, loss = "binary_crossentropy")
    }
  ),
  private = list(
    latent_dim = NA,
    discriminator = NA,
    generator = NA,
    gan_input = NA,
    gan_output = NA,
    optimiser = NA
  )
)




