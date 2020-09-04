import tensorflow as tf
import numpy as np


class ViewSynNet:
    def __init__(self,Disp):
        self.var_dict=[]
        self.filters=32
        self.maxdisp=Disp

    def __call__(self, rgb):
        self.H=rgb.shape[1]
        self.W=rgb.shape[2]
        
        x = self.conv_2d(rgb, 2*self.filters, 'conv.i.1', kernel_size=3, stride=1) 
        x = self.conv_2d(x, 2*self.filters, 'conv.i.2', kernel_size=3, stride=1)
        x = self.conv_2d(x, 2*self.filters, 'conv.i.3', kernel_size=3, stride=2)
        x1 = self.max_pool(x, size=3, stride=2)

        x = self.dense_block(input_x=x1, nb_layers=6, layer_name='dense.1')
        x2 = self.transition_layer(x, scope='trans.1')

        x = self.dense_block(input_x=x2, nb_layers=12, layer_name='dense.2')
        x3 = self.transition_layer(x, scope='trans.2')

        x = self.dense_block(input_x=x3, nb_layers=24, layer_name='dense.3')
        x4 = self.transition_layer(x, scope='trans.3')

        x = self.dense_block(input_x=x4, nb_layers=16, layer_name='dense.o')
        x = self.batch_norm(x, 'linear.bn')
        x = tf.nn.relu(x)
        x = self.global_avg_pool(x)
        x = self.fc_layer(x, self.maxdisp*5*5, 'linear')
        x5=tf.reshape(x,[-1,5,5,self.maxdisp])
        
        with tf.variable_scope("branch", reuse=tf.AUTO_REUSE):
            scale = 1
            branch1_1 = self.conv_2d(x1, self.maxdisp, "conv.1")
            branch1_1 = tf.nn.relu(branch1_1)
            branch1_2 = tf.image.resize_bilinear(branch1_1,[scale * tf.shape(branch1_1)[1],scale * tf.shape(branch1_1)[2]])

            scale *= 2
            branch2_1 = self.conv_2d(x2, self.maxdisp, "conv.2")
            branch2_1 = tf.nn.relu(branch2_1)
            branch2_2 = tf.image.resize_bilinear(branch2_1,[scale * tf.shape(branch2_1)[1],scale * tf.shape(branch2_1)[2]])

            scale *= 2
            branch3_1 = self.conv_2d(x3, self.maxdisp, "conv.3")
            branch3_1 = tf.nn.relu(branch3_1)
            branch3_2 = tf.image.resize_bilinear(branch3_1,[scale * tf.shape(branch3_1)[1],scale * tf.shape(branch3_1)[2]])

            scale *= 2
            branch4_1 = self.conv_2d(x4, self.maxdisp, "conv.4")
            branch4_1 = tf.nn.relu(branch4_1)
            branch4_2 = tf.image.resize_bilinear(branch4_1,[scale * tf.shape(branch4_1)[1],scale * tf.shape(branch4_1)[2]])

            scale *= 2
            branch5_1 = self.conv_2d(x5, self.maxdisp, "conv.5")
            branch5_1 = tf.nn.relu(branch5_1)
            branch5_2 = tf.image.resize_bilinear(branch5_1,[scale * tf.shape(branch5_1)[1],scale * tf.shape(branch5_1)[2]])

            up_sum = branch1_2 + branch2_2 + branch3_2 + branch4_2 + branch5_2
            scale = 4
            up = tf.image.resize_bilinear(up_sum,[scale * tf.shape(up_sum)[1], scale * tf.shape(up_sum)[2]])
            up_conv = self.conv_2d(up, self.maxdisp, "conv.up")

        mask = tf.nn.softmax(up_conv)  
        prob  = self.select(mask, rgb)
        return prob

    
    def select(self, masks, left_image):
        padded = tf.pad(left_image, [[0,0],[0,0],[0, self.maxdisp-1],[0,0]], mode='REFLECT')
        layers = []
        for s in np.arange(self.maxdisp): 
            layers.append(tf.slice(padded, [0,0,s,0], [-1,self.H,self.W,-1]))
        slices = tf.stack(layers, axis=4)
        disparity_image = tf.multiply(slices, tf.expand_dims(masks, axis=3))
        return tf.reduce_sum(disparity_image, axis=4)

    def cost(self,pred,gt):
        mask = gt > 0
        count = 1 if tf.count_nonzero(mask) == 0 else tf.count_nonzero(mask)
        loss=tf.sqrt(tf.pow(gt - pred, 2) + 4.0) /2.0 - 1.0
        loss=tf.reduce_sum(loss*tf.cast(mask,tf.float32), [0,1,2])
        loss /= tf.cast(count,tf.float32)
        return loss


#   Blocks

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope='bottle.i')
            layers_concat.append(x)
            for i in range(nb_layers - 1):
                x = tf.concat(layers_concat, axis=3)
                x = self.bottleneck_layer(x, scope='bottle.' + str(i + 1))
                layers_concat.append(x)
            x = tf.concat(layers_concat, axis=3)

            return x
        
    def bottleneck_layer(self, x, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = self.batch_norm(x, 'bn1')
            x = tf.nn.relu(x)
            x = self.conv_2d(x, 4*self.filters, kernel_size=1, name='conv1')
            x = tf.nn.dropout(x, rate=0.2)

            x = self.batch_norm(x, 'bn2')
            x = tf.nn.relu(x)
            x = self.conv_2d(x, self.filters, kernel_size=3, name='conv2')
            x = tf.nn.dropout(x, rate=0.2)
            
            return x
        
    def transition_layer(self, x, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = self.batch_norm(x,'bn1')
            x = tf.nn.relu(x)
            x = self.conv_2d(x, self.filters//2, kernel_size=1, name='conv1')
            x = tf.nn.dropout(x, rate=0.2)
            x = self.avg_pool(x)

            return x

#   Layers   

    def max_pool(self, x, size=2, stride=2, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding)

    def avg_pool(self, x, size=2, stride=2, padding='SAME'):
        return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding)

    def global_avg_pool(self, x, stride=1) :
        H = x.get_shape().as_list()[1]
        W = x.get_shape().as_list()[2]
        return tf.nn.avg_pool(x, ksize=[1, H, W, 1], strides=[1, stride, stride, 1], padding='SAME')

    def batch_norm(self, inputs, name, epsilon=1e-05, momentum=0.99, trainable=1):
        inputs_shape = inputs.get_shape().as_list()
        params_shape = inputs_shape[-1:]
        beta,gamma = self.get_bn_var(inputs, params_shape, name, trainable)
        axis = list(range(len(inputs_shape) - 1))
        mean, variance = tf.nn.moments(inputs, axes=axis)
        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

    def conv_2d(self, x, out_channels, name, kernel_size=3, stride=1, trainable=1):
        if isinstance(x,np.ndarray):
            in_channels=x.shape[-1]
        else:
            in_channels=x.get_shape().as_list()[-1]
        kernel, biases = self.get_conv_var(kernel_size, in_channels, out_channels, name, trainable)
        x = tf.nn.conv2d(x, kernel,strides=[1,stride,stride,1],padding='SAME')
        x = tf.nn.bias_add(x, biases)     
        return x

    def atrous_conv(self, x, out_channels, name, rate, trainable=1, kernel_size=3):
        in_channels=x.get_shape().as_list()[-1]
        kernel, biases = self.get_conv_var(kernel_size, in_channels, out_channels, name, trainable)
        x = tf.nn.atrous_conv2d(x, kernel,rate=rate,padding='SAME')
        x = tf.nn.bias_add(x, biases)
        return x

    def fc_layer(self, x, out_channels, name, trainable=1):
        input_dims=x.get_shape().as_list()
        in_channels=1
        for i in range(1,len(input_dims)):
            in_channels*=input_dims[i]
        weights, biases = self.get_fc_var(in_channels, out_channels, name, trainable)
        x = tf.reshape(x, [-1, in_channels])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc
    
#   Params

    def _new_variable(self, initial_value, name, idx, var_name, trainable):
        var = tf.Variable(initial_value, name=var_name, trainable=trainable)
        self.var_dict.append(var)
        return var

    def get_bn_var(self, x, params_shape, name, Trainable):     
        initial_value = tf.truncated_normal(params_shape, 0.0, 0.01)
        beta = self._new_variable(initial_value, name, 0, name + "/beta", Trainable)   
        initial_value = tf.truncated_normal(params_shape, 0.0, 0.01)
        gamma = self._new_variable(initial_value, name, 1, name + "/gamma", Trainable)
        return beta, gamma

    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
        filters = self._new_variable(initial_value, name, 0, name + "/filters", trainable)
        initial_value = tf.truncated_normal([out_channels], 0.0, 0.01)
        biases = self._new_variable(initial_value, name, 1, name + "/biases", trainable)
        return filters, biases

    def get_fc_var(self, in_size, out_size, name, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        weights = self._new_variable(initial_value, name, 0, name + "/weights", trainable)
        initial_value = tf.truncated_normal([out_size], 0.0, 0.01)
        biases = self._new_variable(initial_value, name, 1, name + "/biases", trainable)
        return weights, biases
    
    def get_var(self):
        return self.var_dict

