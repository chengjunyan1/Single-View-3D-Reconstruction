import os,sys,random
import numpy as np
import models.end2end as e2e
import models.modeltools as mt
import data.datatools as dt
import tensorflow as tf
import multiprocessing as mp
from PIL import Image

model_path=sys.path[0]
data_path=model_path+'/data/ShapeNetCore.v2.Data'
smn_path=model_path+'/net_params/smn/smn'
pcn_path=model_path+'/net_params/pcn/pcn'
smc_path=model_path+'/net_params/smc/smc'
left_dir=data_path+'/Left_img'
dep_dir=data_path+'/Left_dep'
gt_dir=data_path+'/gt'

def get_batch(step,path):
    data_batch=[]
    l=len(path)
    start=(step*batch_size)%l
    end=((step+1)*batch_size)%l
    if start>end:
        prange=path[start:]
        prange.extend(path[:end])
    else:
        prange=path[start:end]
    for i in prange:
        if path==dep_path or path==dep_path_all: 
            data=np.load(i)
        elif path==left_path or path==left_path_all:
            data=np.array(Image.open(i))/255.0
        elif path==gt_path:
            data=np.load(i)
        data_batch.append(data)
    data_batch=np.array(data_batch).astype(np.float32)
    return data_batch

id_list=dt.get_ids(data_path)
random.shuffle(id_list)
left_path=[os.path.join(left_dir,i+'.png') for i in id_list]
dep_path=[os.path.join(dep_dir,i+'.npy') for i in id_list]
gt_path=[os.path.join(gt_dir,i+'.npy') for i in id_list]

all_ids=[i.split('.')[0] for i in os.listdir(left_dir)]
random.shuffle(all_ids)
left_path_all=[os.path.join(left_dir,i+'.png') for i in all_ids]
dep_path_all=[os.path.join(dep_dir,i+'.npy') for i in all_ids]

batch_size=2
smc_step=99999
smn_step=50000
pcn_step=50000

def train_smn():    
    lr = 0.00005
    num_epochs = 5
    num_iter = len(dep_path) * num_epochs//batch_size
    print("Number of Training Iterations : " + str(num_iter))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
    config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
    config.allow_soft_placement=True
    sess = tf.Session(config=config)
    
    # Building Net based on VGG weights 
    inputs_left = tf.placeholder(tf.float32, [batch_size, 160, 160, 3], name='input_left')
    gt = tf.placeholder(tf.float32, [batch_size, 160, 160] , name='ground_truth')
    global_step = tf.Variable(0, trainable=False, name='global_step')    
    
    net = e2e.SMNet()
    #is_training=tf.constant(True,dtype=tf.bool)
    out=net(inputs_left)
    cost = net.cost(out,gt)
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    # Run initializer 
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    saver = tf.train.Saver(net.vars(),max_to_keep=1)
    if os.path.exists(smn_path+'-'+str(smn_step)+'.meta'):
        saver.restore(sess, smn_path+'-'+str(smn_step))

    print(">> Start training <<")
    print_step = 1
    save_step = 1000
    print("saving every " + str(save_step) + " iterations")
    # ---------- Training Loop --------------- #
    step = sess.run(global_step)
    try:
        while not coord.should_stop():
            # Traing Step
            left_batch=get_batch(step,left_path_all)
            gt_batch=get_batch(step,dep_path_all)
            _, cost_val = sess.run([train, cost], 
                                    feed_dict={inputs_left: left_batch,
                                               gt: gt_batch})
            if step%print_step == 0:
                print(str(step).ljust(10) + ' | Cost: ' + str(cost_val).ljust(10))
            if step%save_step == 0 and step!=0:
                print("Checkpoint Save")
                saver.save(sess, smn_path, step)
            step+=1
            if step==50000:
                break     
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    
    # Store Final Traing Output
    print("\nTraining Completed, storing weights")
    #saver.save(sess, smn_path, step)                                
    coord.join(queue_threads)
    sess.close()

def train_pcn():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
    config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
    config.allow_soft_placement=True
    sess = tf.Session(config=config)

    # Building Net based on VGG weights 
    inputs = tf.placeholder(tf.float32, [batch_size, 160, 160] , name='inputs')
    gt = tf.placeholder(tf.float32, [batch_size, 16384, 3] , name='ground_truth')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    beta = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'beta_op')

    net = e2e.PCNet()
    out=net(inputs)
    cost = net.cost(out[0],out[1],gt,alpha,beta)

    lr = tf.train.exponential_decay(0.0001, global_step,50000, 0.7,staircase=True, name='lr')
    lr = tf.maximum(lr, 1e-6)
    trainer = tf.train.AdamOptimizer(lr)
    train_op = trainer.minimize(cost, global_step)
    # Run initializer 
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    pcn_var=tf.trainable_variables()
    saver_pcn = tf.train.Saver(pcn_var,max_to_keep=1)
    if os.path.exists(pcn_path+'-'+str(pcn_step)+'.meta'):
        saver_pcn.restore(sess, pcn_path+'-'+str(pcn_step))
        print('pcn model loaded')
    else:
        mt.init_net('pcn',model_path,sess)
    print(">> Start training <<")
    print_step = 1
    save_step = 1000
    print("saving every " + str(save_step) + " iterations")
    # ---------- Training Loop --------------- #
    step = sess.run(global_step)
    try:
        while not coord.should_stop():
            gt_batch=get_batch(step,gt_path)
            dep_batch=get_batch(step,dep_path)
            _, cost_val = sess.run([train_op, cost],
                                    feed_dict={inputs: dep_batch,
                                                gt: gt_batch})
            if step%print_step == 0:
                print(str(step).ljust(10) + ' | Cost: ' + str(cost_val).ljust(10))
            if step%save_step == 0 and step!=0:
                print("Checkpoint Save")
                saver_pcn.save(sess, pcn_path, step)
            step+=1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    # Store Final Traing Output
    print("\nTraining Completed, storing weights")
    saver_pcn.save(sess, pcn_path, step)
    coord.join(queue_threads)
    sess.close()

#//_____________________________SMC END2END_________________________________

def train_smc():
    num_epochs = 5
    num_iter = len(dep_path) * num_epochs//batch_size
    print("Number of Training Iterations : " + str(num_iter))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
    config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
    config.allow_soft_placement=True
    sess = tf.Session(config=config)
    
    # Building Net based on VGG weights 
    inputs_left = tf.placeholder(tf.float32, [batch_size, 160, 160, 3], name='input_left')
    gt = tf.placeholder(tf.float32, [batch_size, 16384, 3] , name='ground_truth')
    mask = tf.placeholder(tf.float32, [batch_size, 160, 160] , name='dep_mask')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    beta = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'beta_op')    

    net = e2e.PCNet()
    out=net(mask)
    pcn_var=tf.trainable_variables()

    net = e2e.SMCNet()
    out=net(inputs_left,mask)    
    cost = net.cost(out[0],out[1],gt,alpha,beta)

    lr = tf.train.exponential_decay(0.0001, global_step,50000, 0.7,staircase=True, name='lr')
    lr = tf.maximum(lr, 1e-6)
    trainer = tf.train.AdamOptimizer(lr)
    train_op = trainer.minimize(cost, global_step)
    # Run initializer 
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    smn_var=net.smn.vars()
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    saver_smn = tf.train.Saver(smn_var)
    saver_pcn = tf.train.Saver(pcn_var)
    saver = tf.train.Saver(smn_var+pcn_var,max_to_keep=1)

    if os.path.exists(smc_path+'-'+str(smc_step)+'.meta'):
        saver.restore(sess, smc_path+'-'+str(smc_step))
        print('smc model loaded')
    else:
        if os.path.exists(smn_path+'-'+str(smn_step)+'.meta'):
            saver_smn.restore(sess, smn_path+'-'+str(smn_step))
            print('smn model loaded')
        if os.path.exists(pcn_path+'-'+str(pcn_step)+'.meta'):
            saver_pcn.restore(sess, pcn_path+'-'+str(pcn_step))
            print('pcn model loaded')
    print(">> Start training <<")
    print_step = 1
    save_step = 1000
    print("saving every " + str(save_step) + " iterations")
    # ---------- Training Loop --------------- #
    step = sess.run(global_step)
    try:
        while not coord.should_stop():
            left_batch=get_batch(step,left_path)
            gt_batch=get_batch(step,gt_path)
            mask_batch=get_batch(step,dep_path)
            dep_mask=mt.make_mask(mask_batch)
            _, cost_val = sess.run([train_op, cost],
                                    feed_dict={inputs_left: left_batch,
                                                mask:dep_mask,
                                                gt: gt_batch})
            if step%print_step == 0:
                print(str(step).ljust(10) + ' | Cost: ' + str(cost_val).ljust(10))
            if step%save_step == 0 and step!=0:
                print("Checkpoint Save")
                saver.save(sess, smc_path, step)
            step+=1     
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    
    # Store Final Traing Output
    print("\nTraining Completed, storing weights")
    saver.save(sess, smc_path, step)                
    coord.join(queue_threads)
    sess.close()

#train_smn()
