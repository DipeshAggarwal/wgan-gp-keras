from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow as tf

class WGAN_GP(Model):
    
    def __init__(self, critic, generator, latent_dim, critic_extra_steps, gp_weight=10.0):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_extra_steps = critic_extra_steps
        self.gp_weight = gp_weight
        
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        
    def compile(self, d_optimiser, g_optimiser, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimiser = d_optimiser
        self.g_optimiser = g_optimiser
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        
        interpolated = real_images +alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
            
        # Calculate the gradient w.r.t the interpolated image
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gradient_penalty
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Train the critic more often than generator
        for i in range(self.critic_extra_steps):
            with tf.GradientTape() as tape:
                pred_real = self.critic(real_images, training=True)
                fake_images = self.generator(noise, training=True)
                pred_fake = self.critic(fake_images, training=True)
                
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                
                d_loss = self.d_loss_fn(pred_real, pred_fake) + gp * self.gp_weight
                
            # Compute critic gradients
            grads = tape.gradient(d_loss, self.critic.trainable_variables)
            
            # Update critic weights
            self.d_optimiser.apply_gradients(zip(grads, self.critic.trainable_variables))
            
        # Train the generator
        misleading_labels = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            pred_fake = self.critic(fake_images, training=True)
            g_loss = self.g_loss_fn(pred_fake)
            
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }
        
