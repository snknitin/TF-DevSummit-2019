# TF-DevSummit-2019
Notes from the dev summit


## Code tips


* tf.function and autograph
  
      @tf.function
      def f(x):
        while tf.reduce_sum(x)>1:
          x = tf.tanh(x)
        return x

      f(tf.random.uniform([10]))
      
* 
