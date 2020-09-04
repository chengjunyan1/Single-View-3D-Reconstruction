import tensorflow as tf
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
#from pc_distance import tf_nndistance, tf_approxmatch


def init_net(net,model_path,sess=None):
    if net=='pcn':
        path=model_path+'/net_params/pcn_cd'
        print('loading pcn...')
        saver = tf.train.Saver()
        saver.restore(sess, path)
    elif net=='smn':
        vgg_dict=np.load(os.path.join(model_path,'net_params/vgg19.npy'), encoding='latin1').item()
        print('loading vgg19...')
        del vgg_dict[u'fc6']
        del vgg_dict[u'fc7']
        del vgg_dict[u'fc8']
        return vgg_dict

def dm2pc(dm,shape,camera=[7.314285755157471,0,0,320,180]): # dm:[N H W] camera:[scale,cx,cy,fx,fy]
    N=int(shape[0])
    H=int(shape[1])
    W=int(shape[2])
    dm=tf.contrib.layers.flatten(dm) # N,H,W to N,H*W
    z=dm/camera[0]
    x=np.arange(W) - camera[1]
    x=np.tile(np.tile(x,H),N).reshape(N,H*W) * z / camera[3] 
    y=np.arange(H) - camera[2]
    y=np.tile(y.repeat(W),N).reshape(N,H*W) * z / camera[4] 
    pcs=tf.stack([x,y,z],axis=2)
    return pcs
   
def save(sess,path,var_dict):
    data_dict = {}
    for (name, idx), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}
        data_dict[name][idx] = var_out
    np.save(path, data_dict)
    print(("params saved"))

def load(path):
    net_params=np.load(path).item()
    return net_params

def MatrixToImage(data):
    return Image.fromarray(data.astype(np.uint8))

def dep_viewer(inputs):
    if isinstance(inputs,str):
        x=np.load(inputs)
    else:
        x=inputs
    if len(x.shape)==3:
        x=x[0]
    cv2.imshow("depth_map",x/5.0)
    cv2.waitKey(0)

def make_mask(dep):
    return dep!=0

def resample_pcd(pcd, n):
    sampler=tf.cond(tf.less_equal(tf.shape(pcd)[0],0),
        lambda:tf.constant(np.zeros([n,3]).astype(np.float32)),
        lambda:tf.gather(pcd, tf.random_uniform([n], minval=0,maxval=tf.shape(pcd)[0],dtype=tf.int32)))
    return sampler

def create_pcd(dep_batch,mask,input_points=1024):
    N,H,W=mask.shape
    dep_batch*=mask
    pcd=dm2pc(dep_batch,[N,H,W])
    pcd_batch=[]
    for i in range(N):
        term=pcd[i]
        intermediate_tensor = tf.reduce_sum(tf.abs(term), 1)
        zero_vector = tf.zeros(shape=(1), dtype=tf.float32)
        bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        omit_zeros = tf.boolean_mask(term, bool_mask)
        resampled=resample_pcd(omit_zeros,input_points)
        pcd_batch.append(resampled)
    return tf.stack(pcd_batch)

def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2

def chamfer_part(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    return tf.reduce_mean(tf.sqrt(dist1))

def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)

