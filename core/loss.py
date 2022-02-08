import tensorflow as tf

def d_wasserstein_loss(pred_real, pred_fake):
    real_loss = tf.reduce_mean(pred_real)
    fake_loss = tf.reduce_mean(pred_fake)
    
    return fake_loss - real_loss

def g_wasserstein_loss(pred_fake):
    return -tf.reduce_mean(pred_fake)
