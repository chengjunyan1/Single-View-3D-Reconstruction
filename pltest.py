smn_step=25000
pcn_step=25000
smc_step=1000

import os,sys,random
import numpy as np
import tensorflow as tf
import numpy as np
import models.synnet as sn
import models.matchnet as mn
import models.completnet as cn
import models.modeltools as mt
import models.end2end as e2e
import data.datatools as dt
from PIL import Image
from functools import reduce
from operator import mul
import time

model_path=sys.path[0]
output_path=model_path+'/test'
data_path='/home/chengjunyan1/smcnet/data/ShapeNetCore.v2.Data'
left_dir=data_path+'/Left_img'
dep_dir=data_path+'/Left_dep'
id_list=dt.get_ids(data_path)
random.shuffle(id_list)
#id_list=['02691156+d22521d217d89f8d5b1bb801ea1e2db7']
id_list=[
'02691156+d22521d217d89f8d5b1bb801ea1e2db7',
'02933112+2b3f95e3a1ae1e3f824662341ce2b233',
'02958343+fd98badd65df71f5abfee5fef2dd81c8',
'03001627+20d01aedf15d4d1d23561140a396280f',
'03636649+3f968096c74ee3a3b04a2e6a78ff6c49',
'04256520+2de1b39fe6dd601f1e03b4418ed1a4f3',
'04379243+1a15e651e26622b0e5c7ea227b17d897',
'04530566+fd850819ad05f139bd4f838682e34d2a'
]

def test_smn():
    inputs_left=[]
    for i in id_list:
        left_path=os.path.join(left_dir,i+'.png')
        temp=np.array(Image.open(left_path)).astype(np.float32)
        inputs_left.append(temp)
    inputs_left=np.stack(inputs_left)

    sess = tf.Session()
    net = e2e.SMNet()
    out=net(inputs_left)[net.r]
    right=net.right

    # Run initializer 
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    saver=tf.train.Saver(net.vars())
    saver.restore(sess,model_path+'/net_params/smn/smn-'+str(smn_step))

    out=sess.run(out)
    right=sess.run(right)
    np.save(os.path.join(output_path,'testsmn.npy'),out)    
    np.save(os.path.join(output_path,'testright.npy'),right)
    #print(id_list[0])

def test_pcn():
    inputs_dep=[]
    for i in id_list:
        dep_path=os.path.join(dep_dir,i+'.npy')
        temp=np.array(np.load(dep_path)).astype(np.float32)
        inputs_dep.append(temp)
    inputs_dep=np.stack(inputs_dep)

    sess = tf.Session()
    net = e2e.PCNet()
    out=net(inputs_dep)
    pcn_var=tf.trainable_variables()
    # Run initializer 
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    saver_pcn = tf.train.Saver(pcn_var,max_to_keep=1)
    saver_pcn.restore(sess, model_path+'/net_params/pcn/pcn-'+str(pcn_step))
    out=sess.run(out)
    np.save(os.path.join(output_path,'testpcn.npy'),out[1])

def test_smc():
    inputs_left=[]
    mask_batch=[]
    for i in id_list:
        left_path=os.path.join(left_dir,i+'.png')
        dep_path=os.path.join(dep_dir,i+'.npy')
        temp=np.array(Image.open(left_path)).astype(np.float32)
        inputs_left.append(temp)
        temp=np.array(np.load(dep_path)).astype(np.float32)
        mask_batch.append(temp)
    inputs_left=np.stack(inputs_left)
    mask_batch=np.stack(mask_batch)

    sess = tf.Session()

    net = e2e.PCNet()
    out=net(mask_batch)
    pcn_var=tf.trainable_variables()

    net = e2e.SMCNet()
    dep_mask=mt.make_mask(mask_batch)
    out=net(inputs_left,dep_mask)
    dep=net.dep
    right=net.right
    # Run initializer 
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

    smn_var=net.smn.vars()
    saver=tf.train.Saver(smn_var+pcn_var)
    saver.restore(sess,model_path+'/net_params/smc/smc-'+str(smc_step))

    out=sess.run(out)
    dep=sess.run(dep)
    right=sess.run(right)
    np.save(os.path.join(output_path,'testsmc.npy'),out[1])
    #np.save(os.path.join(output_path,'testdep.npy'),dep)
    #np.save(os.path.join(output_path,'testright.npy'),right)
    #np.save(os.path.join(output_path,i+'_coarse.npy'),out[0])

test_smc()
#test_pcn()
#test_smn()
