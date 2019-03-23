# TF-DevSummit-2019
Notes from the dev summit


## Installation

    pip install -U --pre tensorflow

You can use a conversion script --> tf_upgrade_v2

    tf_upgrade_v2 --infile file1.py --outfile file2.py

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
* Escape to backwards compatibility module to use some old function or a deprecated funcntionality
    * `tf.compact.v1`(doesn't include tf.contrib)
* Use any keras apis from **tf.keras**
    *  Debugging and Dynamic control with Eager
    *  tf.keras.optimizers.{Adadelta,Adam,Adagrad,Adamax,Nadam,RMSprop,SGD}
    *  tf.keras.metrics, tf.keras.losses, tf.keras.layers
    *  For RNN and LSTM, there is only one version which has been modified to `model.add(tf.keras.layer.LSTM(32))` instead of
            
           if tf.test.is_gpu_available():
                model.add(tf.keras.layers.CudnnLSTM(32))
           else:
                model.add(tf.keras.layer.LSTM(32))
    *   Tensorboard integration can be done in one line
        
            tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            model.fit(x_train,y_train,epochs=5, validation_data = [x_test,y_test], callbacks = [tb_callback])
      
* **Distribute Strategy-Multi GPU** : set of built in strategies for distributing training workflows that work natively with keras. We can distribute the model across multiple gpus by defining the model network, compiling and fitting within the strategy scope. Data will be prefetched to GPUs batch by batch , variables will be mirrorred in sync across all available devices using allreduce and youget greated than 90%  scaling efficiency over multiple GPUs.

       strategy = tf.distribute.MirroredStrategy()
       with strategy.scope():
            model = tf.keras.models.Sequential([...])
            model.compile(.....)
 
* Saving Models and beyond - 

        saved_model_path = tf.keras.experimental.export_saved_model(model, path)
        load_model = tf.keras.experimental.load_from_saved_model(path)
        load_model.summary()
        
* Multi-node synchronous - even TPUStrategy

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.Sequential([...])
            model.compile(.....)
    
