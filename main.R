library(here)
library(purrr)
library(yaml)

# Source code
map(paste("src/",list.files(here("src")), sep=""), source)

# Load settings for script
config <- yaml.load_file(here("settings.yaml"))

# Instantiate GAN
discriminator <- Discriminator$new(config$model$height,
                                  config$model$width,
                                  config$model$channels)

generator <- Generator$new(config$model$latent_dim,
              config$model$channels)

gan <- Gan$new(latent_dim = config$model$latent_dim,
               discriminator$discriminator,
               generator$generator)

# Load data
cifar10 <- dataset_cifar10()

c(c(x_train, y_train), c(x_test, y_test)) %<-% cifar10


x_train <- x_train[as.integer(y_train) == 6, , ,]
x_train <- x_train / 255

# Create directory if it doesn't exist
if (!any(config$output$dir == list.files(here()))) {
  dir.create(path=here(config$output$dir))
}

# Train GAN
for (step in 1:config$model$iterations) {
  
  random_latent_vectors <- matrix(rnorm(config$model$batch_size * 
                                        config$model$latent_dim),
                                  nrow = config$model$batch_size, 
                                  ncol = config$model$latent_dim)
  
  generated_images <- generator$generator %>% predict(random_latent_vectors)
  
  stop <- config$model$start + batch_size - 1
  real_images <- x_train[start:stop, , ,]
  rows <- nrow(real_images)
  combined_images <- array(0, dim = c(rows * 2, dim(real_images)[-1]))
  combined_images[1:rows,,,] <- generated_images
  combined_images[(rows+1) : (rows*2),,,] <- real_images  
  
  labels <- rbind(matrix(1, nrow = config$model$batch_size, ncol=1),
                  matrix(0, nrow = config$model$batch_size, ncol=1))
  
  labels <- labels + (0.5 * array(runif(prod(dim(labels))),
                                  dim = dim(labels)))
  
  d_loss <- discriminator$discriminator %>% 
    train_on_batch(combined_images, labels)
  
  random_latent_vectors <- matrix(rnorm(config$model$batch_size * 
                                        config$model$latent_dim),
                                  nrow = config$model$batch_size,
                                  ncol = config$latent_dim)
  
  misleading_targets <- array(0, dim = c(config$model$batch_size, 1))
  
  a_loss <- gan$gan %>% 
    train_on_batch(random_latent_vectors, misleading_targets)
  
  start <- start + config$model$batch_size
  
  if (start > (nrow(x_train) - config$model$batch_size))
      start <- 1
  
  if (step %% 100 == 0) {
    
    save_model_weights_hdf5(gan, here(config$output$dir,"gan.h5"))
    cat("discriminator loss:", d_loss, "\n")
    cat("adversarial_loss:", a_loss, "\n")
    
    image_array_save(
      generated_images[1,,,] * 255,
      path = here(config$output$dir,paste0("generated_frog", step, ".png"))
    )
    
    image_array_save(
      real_images[1,,,] * 255,
      path = here(config$output$dir,paste0("real_frog", step, ".png"))
    )
  }
}

