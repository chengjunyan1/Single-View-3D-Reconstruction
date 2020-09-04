import bpy,math,os,copy,mathutils,random

C = bpy.context
D = bpy.data
scene = D.scenes['Scene']
cam = D.objects['Camera']
lamp = D.objects['Lamp']
render_setting = scene.render
w = 160
h = 160
render_setting.resolution_x = w*2
render_setting.resolution_y = h*2
cam_angle=(70,45)
obj_vecs=(0.0449,-0.0449,0.0) #6.35cm
angles=[
    (0,0,0),(0,0,90),
    (0,0,180),(0,0,270),
    (0,180,0),(0,180,90),
    (0,180,180),(0,180,270),
]

def set_cam():
    center=(0,0,0)
    if 'center' in D.objects:
        D.objects['center'].select=True
        bpy.ops.object.delete()
    bpy.ops.object.empty_add(type='SPHERE')
    D.objects['Empty'].name = 'center'
    focus = D.objects['center']
    focus.location = center
    scene.objects.active = cam
    cam.select = True
    if 'Track To' not in cam.constraints:
        bpy.ops.object.constraint_add(type='TRACK_TO')
    cam.constraints['Track To'].target = focus
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'
    r = 2.
    theta=cam_angle[0]*math.pi/180.
    phi=cam_angle[1]*math.pi/180.
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)
    D.objects['Lamp'].location = (loc_x, loc_y, loc_z)
    D.objects['Camera'].location = (loc_x, loc_y, loc_z)

def move_obj():
    for i in D.objects:
        if('mesh' in i.name):
            i.location+=mathutils.Vector(obj_vecs)

def rotate_obj(k):
    for i in D.objects:
        if('mesh' in i.name):
            i.delta_rotation_euler.x=angles[k][0]*math.pi/180.
            i.delta_rotation_euler.y=angles[k][1]*math.pi/180.
            i.delta_rotation_euler.z=angles[k][2]*math.pi/180.

def clear():
    for ob in scene.objects:
        ob.select = True if ob.type == 'MESH' else False
    bpy.ops.object.delete()

########################################################

model_path = 'C:\\ChengJunyan1\\3d\\ShapeNetCore.v2'
image_dir = 'C:\\ChengJunyan1\\3d\\ShapeNetCore.v2.OpenEXR'
common_name='\\models\\model_normalized.obj'
model_num=52491

def load_model(model_path):
    bpy.ops.import_scene.obj(filepath=model_path, filter_glob='*.obj')

def save(name,imgtype):
    path = os.path.join(image_dir,imgtype)
    path = os.path.join(path, name+'.exr')
    D.images['Render Result'].save_render(filepath=path)
    print('save to ' + path)

def all_path():
    ap=[]
    categories=os.listdir(model_path)
    for i in categories:
        category_path=os.path.join(model_path,i)
        objs=os.listdir(category_path)
        for j in objs:
            path=os.path.join(category_path,j+common_name)
            name=i+'+'+j
            ap.append((path,name))
    return ap

def work_log():
    l_dir=os.listdir(os.path.join(image_dir,'left'))
    r_dir=os.listdir(os.path.join(image_dir,'right'))
    l_log=[i.split('.')[0].split('+')[0]+'+'+i.split('.')[0].split('+')[1] for i in l_dir]
    r_log=[i.split('.')[0].split('+')[0]+'+'+i.split('.')[0].split('+')[1] for i in r_dir]
    return [l_log,r_log]

def capture(work_range=range(0,model_num)):
    set_cam()
    ids=all_path()
    [logl,logr]=work_log()
    for i in work_range:
        clear()
        path=ids[i][0]
        name=ids[i][1]
        if os.path.exists(path):
            rd=random.sample(range(0,8),1)[0]
            sname=name+'+r' if rd in range(4,8) else name
            if name not in logl or name not in logr:
                load_model(path)
                rotate_obj(rd)
                bpy.ops.render.render()
                save(sname,'left')
                print('******** '+str(i)+': '+name+' left finished ********\n')
                move_obj()
                bpy.ops.render.render()
                save(sname,'right')
                print('******** '+str(i)+': '+name+' right finished ********\n\n')

# Set Dist
worker_id=0
work_each=10000
work_range=range(worker_id*work_each,min((worker_id+1)*work_each,model_num))
capture(work_range)