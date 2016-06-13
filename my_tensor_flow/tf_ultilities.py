import tensorflow as tf


class Costs:
    @staticmethod
    def cross_entropy(y, predicted_y):
        return tf.reduce_mean(-tf.reduce_sum(predicted_y * tf.log(y), reduction_indices=[1]))


# def tf_set_value(session, )
