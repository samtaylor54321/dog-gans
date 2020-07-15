library(here)
library(yaml)

# Source code
lapply(paste("src/",list.files(here("src")), sep=""), source)

# Load settings for script
message("Loading Config...")
config <- yaml.load_file(here("settings.yaml"))

# Instantiate GAN
message("Instantiating Discriminator...")
discriminator <- Discriminator$new(config$model$height,
                                  config$model$width,
                                  config$model$channels)

message("Instantiating Generator...")
generator <- Generator$new(config$model$latent_dim,
              config$model$channels)

message("Instantiating GAN...")
gan <- Gan$new(latent_dim = config$model$latent_dim,
               discriminator$discriminator,
               generator$generator)


# Load data
message("Loading Data...")
datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  # This is the target directory
  directory = here(config$input$dir),
  # This is the data generator
  datagen,
  # All images will be resized 
  target_size = config$input$image_shape,
  batch_size = config$model$batch_size,
  class_mode = NULL
)

x_train <- generator_next(train_generator)


# Create directory if it doesn't exist
message("Creating output directory...")
if (!any(config$output$dir == list.files(here()))) {
  dir.create(path=here(config$output$dir))
}

# Train GAN
message("Training GAN...")
for (step in 1:config$model$iterations) {
  
  message(paste0("  Step ", step, " of ", config$model$iterations, "..."))
  
  random_latent_vectors <- matrix(rnorm(config$model$batch_size * 
                                        config$model$latent_dim),
                                  nrow = config$model$batch_size, 
                                  ncol = config$model$latent_dim)
  
  generated_images <- generator$generator %>% predict(random_latent_vectors)
  
  stop <- config$model$start + config$model$batch_size - 1
  real_images <- x_train[config$model$start:stop, , ,]
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
                                  ncol = config$model$latent_dim)
  
  misleading_targets <- array(0, dim = c(config$model$batch_size, 1))
  
  a_loss <- gan$gan %>% 
    train_on_batch(random_latent_vectors, misleading_targets)
  
  start <- config$model$start + config$model$batch_size
  
  if (start > (nrow(x_train) - config$model$batch_size))
      start <- 1
  
  if (step %% 100 == 0) {
    
    message("Saving model and outputs...")
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

