# TF-DevSummit-2019
Notes from the dev summit


## Installation

    pip install -U --pre tensorflow

## Code tips


* **tf.function** and autograph
  
      @tf.function  # Turns eager code into a graph function by function
      def f(x):
        while tf.reduce_sum(x)>1:
          x = tf.tanh(x)
        return x

      f(tf.random.uniform([10]))
      
* **tf.keras** as the high level API with Eager execution as default
* Internal ops are accessible in **tf.raw_ops**
