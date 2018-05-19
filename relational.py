import tensorflow as tf
import numpy as np
from model import RN, Mlp


if __name__ == "__main__":
	rn = RN( Mlp([12,12,12],name="f"), lambda t: tf.reduce_sum( t, axis = 0 ), Mlp([12,12,12],name="g") )
	rn_input = tf.placeholder( tf.float32, shape=[None,12] )
	rn_output = rn( rn_input )
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		for i in range( 2, 17 ):
			print(
				i,
				sess.run(
					rn_output,
					feed_dict = {
						rn_input: np.random.rand( i,12 )
					}
				)
			)
	#end Session
