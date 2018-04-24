import tensorflow as tf
import numpy as np
import random

class Mlp(object):
	def __init__(
		self,
		layer_sizes,
		output_size = None,
		activations = None,
		output_activation = None,
		use_bias = True,
		kernel_initializer = None,
		bias_initializer = tf.zeros_initializer(),
		kernel_regularizer = None,
		bias_regularizer = None,
		activity_regularizer = None,
		kernel_constraint = None,
		bias_constraint = None,
		trainable = True,
		name = None,
		name_internal_layers = True
	):
		"""Stacks len(layer_sizes) dense layers on top of each other, with an additional layer with output_size neurons, if specified."""
		self.layers = []
		internal_name = None
		# If object isn't a list, assume it is a single value that will be repeated for all values
		if not isinstance( activations, list ):
			activations = [ activations for _ in layer_sizes ]
		#end if
		# If there is one specifically for the output, add it to the list of layers to be built
		if output_size is not None:
			layer_sizes = layer_sizes + [output_size]
			activations = activations + [output_activation]
		#end if
		for i, params in enumerate( zip( layer_sizes, activations ) ):
			size, activation = params
			if name_internal_layers:
				internal_name = name + "_MLP_layer_{}".format( i + 1 )
			#end if
			new_layer = tf.layers.Dense(
				size,
				activation = activation,
				use_bias = use_bias,
				kernel_initializer = kernel_initializer,
				bias_initializer = bias_initializer,
				kernel_regularizer = kernel_regularizer,
				bias_regularizer = bias_regularizer,
				activity_regularizer = activity_regularizer,
				kernel_constraint = kernel_constraint,
				bias_constraint = bias_constraint,
				trainable = trainable,
				name = internal_name
			)
			self.layers.append( new_layer )
		#end for
	#end __init__
	
	def __call__( self, inputs, *args, **kwargs ):
		outputs = [ inputs ]
		for layer in self.layers:
			outputs.append( layer( outputs[-1] ) )
		#end for
		return outputs[-1]
	#end __call__
#end Mlp

class RN(object):
	def __init__( self, f, reducer, g ):
		self.__ONE_I = tf.constant(1,dtype=tf.int32)
		self.__ZERO_I = tf.constant(0,dtype=tf.int32)
		self.__TWO_I = tf.constant(2,dtype=tf.int32)
		self.f = f
		self.g = g
		self.r = reducer
	#end __init__
	
	def __call__( self, inputs, *args, **kwargs ):
		# Save invariant values in object to avoid consuming space for being
		# returned from the loop
		self.n = tf.shape( inputs )[0]
		self.n_minus_one = tf.subtract( self.n, self.__ONE_I )
		self.inputs = inputs
		# Guarantee that we have at least two elements on the set
		assert_op = tf.Assert(tf.greater(self.n, 1), [self.n],name="assert_set_has_at_least_two_elements")
		with tf.control_dependencies([assert_op]):
			# Each pair (obj_i, obj_j), except the ones where i == j will be used
			# So (n^2)/2 - n
			number_of_outputs =  tf.floordiv( tf.subtract(tf.multiply( self.n, self.n ), self.n ), self.__TWO_I )
			
			# Define loop functions (has to be defined here to be able to call "global"
			# variables such as self.n, self.n_minus_one and self.inputs
			def processSetLoop(i,j,oi,output):
				# Gather values from inputs
				obj_i = self.inputs
				obj_j = self.inputs
				# Process object pair
				output, oi = tf.cond(
					tf.not_equal( i, j ),
					lambda: ( output.write( oi, self.g( obj_i, obj_j ) ), tf.add( oi, self.__ONE_I ) ),
					lambda: ( output, oi )
				)
				# Increment loop variables
				i = tf.add( i, self.__ONE_I )
				new_i, new_j = tf.cond(
					tf.equal( i, self.n ),
					lambda: ( tf.add( j, self.__ONE_I ), tf.add( j, self.__ONE_I ) ),
					lambda: ( i, j )
				)
				return new_i, new_j, oi, output
			#end processSetLoop
			def processSetCond(i,j,oi,output):
				return tf.less( j, self.n_minus_one )
			#end processSetCond
			
			# Output TensorArray from the set processing
			set_outputs = tf.TensorArray(
				inputs.dtype,
				size = number_of_outputs
			)
			# Process the set of object pairs
			_, _, _, set_outputs = tf.while_loop(
				processSetCond,
				processSetLoop,
				[self.__ONE_I,self.__ZERO_I,self.__ZERO_I,set_outputs]
			)
			set_outputs = set_outputs.stack()
			reduced_output = self.r( set_outputs )
			return self.f( reduced_output )
		#end Control Dependencies
		raise( Exception( "Unreachable code reached. Did you pass a set with only one element to the Relational Network module?" ) )
	#end __call__
#end RN

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
