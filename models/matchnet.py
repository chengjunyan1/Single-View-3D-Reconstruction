import tensorflow as tf
import numpy as np

class StereoMatchNet:
    def __init__(self,Disp):
        self.maxdisp = Disp
        self.k = 3 # or 4
        self.r = 3
        self.feature_channels=32
        self.var_dict=[]

    def __call__(self,left,right):
        N,H,W,C=right.shape
        H=H//pow(2,self.k)
        W=W//pow(2,self.k)
        C=self.feature_channels
        self.batch_size=N
        self.coarse_H=H
        self.coarse_W=W

        disp = (self.maxdisp + 1) // pow(2, self.k)
        with tf.variable_scope("feature", reuse=tf.AUTO_REUSE):
            left_feature = self.feature_network(left)
            right_feature = self.feature_network(right)
            
        with tf.variable_scope("cost_volume", reuse=tf.AUTO_REUSE):
            left_feature=tf.tile(tf.expand_dims(left_feature,1),[1,disp,1,1,1])
            cost_mask=np.zeros([N,disp,H,W,C]).astype(np.float32)
            right_roll=[]
            for i in range(disp):
                right_roll.append(tf.roll(right_feature,shift=i,axis=2))
                cost_mask[:,i,:,i:,:]=1
            right_feature=tf.stack(right_roll,axis=1)
            cost=(left_feature-right_feature)*cost_mask 

        with tf.variable_scope("disp", reuse=tf.AUTO_REUSE):
            for i in range(4):
                cost=self.conv_3d(cost,32,'conv3d.'+str(i))
                cost=self.batch_norm(cost,'bn3d.'+str(i))
                cost=tf.nn.leaky_relu(cost,alpha=0.2)
            cost = self.conv_3d(cost,1,'conv3d.o')
            cost = tf.squeeze(cost, 4)
            pred = tf.nn.softmax(cost, axis=1)
            pred = self.disparity_regression(disp,pred)

        with tf.variable_scope("down_sample", reuse=tf.AUTO_REUSE):
            img_pyramid_list = []
            for i in range(self.r):
                v_size=[int(int(right.shape[1])/pow(2,i)),int(int(right.shape[2])/pow(2,i))]
                img_pyramid_list.append(tf.image.resize_bilinear(left,v_size))
            img_pyramid_list.reverse()

        pred_pyramid_list = [pred]
        for i in range(self.r):
            with tf.variable_scope("refine."+str(i), reuse=tf.AUTO_REUSE):
                pred_pyramid_list.append(self.hierarchical_refinement(
                            pred_pyramid_list[i], img_pyramid_list[i]))

        with tf.variable_scope("up_sample", reuse=tf.AUTO_REUSE):
            for i in range(self.r):
                pred_pyramid_list[i] = pred_pyramid_list[i]*(int(right.shape[2])
                        /int(pred_pyramid_list[i].shape[2]))
                pred_pyramid_list[i] = tf.squeeze(tf.image.resize_bilinear(
                        tf.expand_dims(pred_pyramid_list[i], axis=3),
                        [int(i) for i in list(right.shape[1:3])]),axis=3)

        return pred_pyramid_list


    def feature_network(self, x):
        for i in range(self.k):
            x = self.conv_2d(x,32,'conv.'+str(i)+'.0',kernel_size=3,stride=1)#5x5改3x3
            x = self.conv_2d(x,32,'conv.'+str(i)+'.1',kernel_size=3,stride=2)
        for i in range(6):
            inputs=x
            x = self.conv_2d(x,32,'fn.res.'+str(i))
            x = self.batch_norm(x,'fn.bn.'+str(i))
            x = tf.nn.leaky_relu(x,alpha=0.2)
            x = x + inputs
        x=self.conv_2d(x,self.feature_channels,'conv.o')
        return x

    def hierarchical_refinement(self, low_disparity, corresponding_rgb):
        output=tf.expand_dims(low_disparity,axis=3)
        twice_disparity=tf.image.resize_bilinear(output,corresponding_rgb.shape[1:3])
        if(int(corresponding_rgb.shape[2]) / int(low_disparity.shape[2]) >= 1.5):
            twice_disparity *= 2
        output=tf.concat([twice_disparity, corresponding_rgb],3)
        output=self.conv_2d(output,32,'hr.conv.i')
        output = self.batch_norm(output,'hr.bn.i')
        output = tf.nn.leaky_relu(output,alpha=0.2)
        atrous_list = [1, 2, 4, 8, 1, 1]
        k=0
        for i in atrous_list:
            inputs = output
            output = self.atrous_conv(output,32,'hr.atrous.'+str(k),i)
            output = self.batch_norm(output,'hr.bn.'+str(k))
            k+=1
            output = tf.nn.leaky_relu(output,alpha=0.2)
            output = output + inputs
        output=self.conv_2d(output,1,'hr.conv.o')
        output=tf.squeeze(twice_disparity+output,3)
        output=tf.nn.relu(output)
        return output

    def disparity_regression(self,maxdisp,x):
        disp = np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])
        disp = np.repeat(disp, self.batch_size,axis=0)
        disp = np.repeat(disp, self.coarse_H,axis=2)
        disp = np.repeat(disp, self.coarse_W,axis=3)
        disp=tf.Variable(disp.astype(np.float32),name='dr',trainable=False)
        out = tf.reduce_sum(disp*x,axis=1)
        return out
    
    def cost(self,pred,gt):
        mask = gt > 0 
        count = 1 if tf.count_nonzero(mask) == 0 else tf.count_nonzero(mask)
        gt_mean=tf.reduce_sum(gt)/tf.cast(count,tf.float32) #normalization
        gt=gt if gt_mean==0 else gt/gt_mean
        loss=0 
        for i in pred:
            counti = 1 if tf.count_nonzero(i) == 0 else tf.count_nonzero(i)
            i_mean=tf.reduce_sum(i)/tf.cast(counti,tf.float32) #normalization
            i=i if i_mean==0 else i/i_mean
            temp=tf.sqrt(tf.pow(gt - i, 2) + 4.0) /2.0 - 1.0
            temp=tf.reduce_sum(temp*tf.cast(mask,tf.float32), [0,1,2])
            loss+=temp 
        loss /= tf.cast(count,tf.float32)
        return loss


#   Layers

    def batch_norm(self, inputs, name, epsilon=1e-05, momentum=0.99, trainable=1):
        inputs_shape = inputs.get_shape().as_list()
        params_shape = inputs_shape[-1:]
        beta,gamma = self.get_bn_var(inputs, params_shape, name, trainable)
        axis = list(range(len(inputs_shape) - 1))
        mean, variance = tf.nn.moments(inputs, axes=axis)
        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

    def conv_3d(self, x, out_channels, name, trainable=1, kernel_size=3, stride=1):
        in_channels=x.get_shape().as_list()[-1]
        kernel, biases = self.get_3d_var(kernel_size, in_channels, out_channels, name, trainable)
        x=tf.nn.conv3d(x,kernel,[1, stride, stride, stride, 1],padding='SAME')
        x = tf.nn.bias_add(x, biases)      
        return x

    def conv_2d(self, x, out_channels, name, trainable=1, kernel_size=3, stride=1):
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

#   Params

    def _new_variable(self, shape, name, idx, var_name, Trainable):
        var = tf.get_variable(var_name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1),trainable=Trainable)
        if var not in self.var_dict:
            self.var_dict.append(var)
        return var  

    def get_bn_var(self, bottom, params_shape, name, Trainable):       
        beta = self._new_variable(params_shape, name, 0, name + "/beta", Trainable)
        gamma = self._new_variable(params_shape, name, 1, name + "/gamma", Trainable)
        return beta, gamma

    def get_conv_var(self, filter_size, in_channels, out_channels, name, Trainable):
        filters = self._new_variable([filter_size, filter_size, in_channels, out_channels], name, 0, name + "/filters", Trainable)
        biases = self._new_variable([out_channels], name, 1, name + "/biases", Trainable)
        return filters, biases

    def get_3d_var(self, filter_size, in_channels, out_channels, name, Trainable):
        filters = self._new_variable([filter_size, filter_size, filter_size, in_channels, out_channels], name, 0, name + "/filters", Trainable)
        biases = self._new_variable([out_channels], name, 1, name + "/biases", Trainable)
        return filters, biases

    def get_var(self):
        return self.var_dict
