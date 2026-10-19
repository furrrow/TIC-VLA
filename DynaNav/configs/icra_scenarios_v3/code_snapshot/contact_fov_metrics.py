"""Opt-in physical human bodies, PhysX contact reports, and cropped-FOV clearance.
Human capsules: radius0.25m, total height1.8m. Clearance to the robot's convex
collision footprint; no occlusion test. Collision is a solver contact, not distance.
"""
import os,json,math,time
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,PhysicsSchemaTools,Gf
import omni.usd,omni.physx

RADIUS=.25
HEIGHT=1.8

def intersects_crop(points,world_from_camera,projection):
    camera=np.c_[points,np.ones(len(points))] @ np.linalg.inv(np.asarray(world_from_camera))
    front=camera[:,2]<0
    if not front.any():return False
    clip=camera[front] @ np.asarray(projection)
    valid=clip[:,3]>1e-8
    if not valid.any():return False
    ndc=clip[valid,:2]/clip[valid,3,None]
    # Native1920x1080 -> centered1080x1080: horizontal NDC [-.5625,.5625].
    return bool(ndc[:,0].max()>=-.5625 and ndc[:,0].min()<=.5625 and ndc[:,1].max()>=-1 and ndc[:,1].min()<=1)

def polygon_circle_gap(poly,center,radius):
    a=poly;b=np.roll(poly,-1,axis=0);edge=b-a;v=center-a
    t=np.clip(np.sum(v*edge,axis=1)/np.maximum(np.sum(edge*edge,axis=1),1e-12),0,1)
    dist=float(np.linalg.norm(center-(a+t[:,None]*edge),axis=1).min())
    cross=edge[:,0]*v[:,1]-edge[:,1]*v[:,0]
    inside=bool(np.all(cross>=-1e-9) or np.all(cross<=1e-9))
    return (-dist if inside else dist)-radius

class ContactFOVMetrics:
 def __init__(self,stage,robot_path,expected_humans=4):
    self.stage=stage;self.robot_path=robot_path;self.frame=0;self.active=False
    self.human_collision=False;self.obstacle_collision=False;self.contact_pairs=set();self.min_clearance=None;self.fov_frames=0;self.frames=0
    self.root=Path(os.environ['VEGA_TRACE_ROOT'])/os.environ['TICVLA_EPISODE_NAME'];self.root.mkdir(parents=True,exist_ok=True)
    self.humans=[];self.robot_meshes=[]
    self.measure_compute_seconds=0.0
    self._fov_stream=(self.root/'fov_surface_clearance.jsonl').open('a',buffering=65536)
    self._contact_stream=(self.root/'physical_contact_events.jsonl').open('a',buffering=8192)
    parent=stage.GetPrimAtPath('/World/Characters')
    for char in parent.GetChildren():
     if not char.GetName().startswith('Character'):continue
     skel=next((p for p in Usd.PrimRange(char) if p.GetTypeName()=='SkelRoot'),None)
     if skel is None:continue
     path='/World/MetricHumanBodies/'+char.GetName();capsule=UsdGeom.Capsule.Define(stage,path)
     capsule.CreateRadiusAttr(RADIUS);capsule.CreateHeightAttr(HEIGHT-2*RADIUS);capsule.CreateAxisAttr('Z');capsule.CreateVisibilityAttr('invisible')
     capsule.AddTranslateOp();UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
     body=UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim());body.CreateKinematicEnabledAttr(True)
     PhysxSchema.PhysxContactReportAPI.Apply(capsule.GetPrim()).CreateThresholdAttr(0.)
     self.humans.append((skel,capsule))
    if len(self.humans)!=expected_humans:raise RuntimeError(f'Expected {expected_humans} human bodies, got {len(self.humans)}')
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
     if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.IsInstanceProxy():PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
     if prim.GetTypeName() not in ['Mesh','Cylinder','Cube']:continue
     ancestor=prim;enabled=False
     while ancestor and str(ancestor.GetPath()).startswith(robot_path):
      if ancestor.HasAPI(UsdPhysics.CollisionAPI):
       enabled=UsdPhysics.CollisionAPI(ancestor).GetCollisionEnabledAttr().Get() is not False;break
      ancestor=ancestor.GetParent()
     if not enabled:continue
     if prim.GetTypeName()=='Mesh':pts=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
     elif prim.GetTypeName()=='Cube':
      half=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2;pts=np.array([[x,y,z] for x in [-half,half] for y in [-half,half] for z in [-half,half]])
     else:
      cylinder=UsdGeom.Cylinder(prim);radius=float(cylinder.GetRadiusAttr().Get());height=float(cylinder.GetHeightAttr().Get());axis=str(cylinder.GetAxisAttr().Get())
      angles=np.linspace(0,2*np.pi,128,endpoint=False);pts=np.array([[radius*np.cos(t),radius*np.sin(t),z] for t in angles for z in [-height/2,height/2]])
      if axis=='X':pts=pts[:,[2,0,1]]
      elif axis=='Y':pts=pts[:,[0,2,1]]
     if len(pts):
      # Interior vertices cannot change a convex footprint under a rigid transform.
      if len(pts)>32:
       from scipy.spatial import ConvexHull
       pts=pts[ConvexHull(pts,qhull_options='QJ').vertices]
      self.robot_meshes.append((prim,pts))
    if not self.robot_meshes:raise RuntimeError('No robot collision geometry; cannot measure footprint clearance')
    self.sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)
    self.sync()
    print('CONTACT_FOV_READY',json.dumps({'humans':len(self.humans),'robot_collision_shapes':len(self.robot_meshes),'radius':RADIUS,'height':HEIGHT}),flush=True)
 def sync(self):
    for skel,capsule in self.humans:
     pos=UsdGeom.Xformable(skel).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
     capsule.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(pos[0],pos[1],pos[2]+HEIGHT/2))
 def on_contacts(self,headers,data):
    if not self.active:return
    for h in headers:
     actor=[str(PhysicsSchemaTools.intToSdfPath(x)) for x in [h.actor0,h.actor1]]
     robot=[x.startswith(self.robot_path) for x in actor]
     if sum(robot)!=1:continue
     other=actor[1] if robot[0] else actor[0]
     points=[data[i] for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data)]
     # Contact manifolds can exist at a positive contact offset; require touching/penetration.
     touching=[p for p in points if p.separation<=0.0]
     if not touching:continue
     human=other.startswith('/World/MetricHumanBodies/')
     # Human pairs always count. Other pairs count on chassis, excluding expected wheel support contacts.
     robot_collider=str(PhysicsSchemaTools.intToSdfPath(h.collider0 if robot[0] else h.collider1))
     support=any(t in robot_collider.lower() for t in ['wheel','caster'])
     if not human and support and all(abs(p.normal[2])>.9 and p.position[2]<.05 for p in touching):continue
     self.human_collision |= human;self.obstacle_collision |= not human
     event={'frame':self.frame,'actors':actor,'robot_collider':robot_collider,'human':human,'separations_m':[float(p.separation) for p in touching],'positions':[list(p.position) for p in touching],'impulses':[list(p.impulse) for p in touching]}
     self._contact_stream.write(json.dumps(event)+'\n');self._contact_stream.flush()
     self.contact_pairs.add(tuple(sorted(actor)))
 def measure(self,frame):
    from scipy.spatial import ConvexHull
    measure_started=time.perf_counter()
    self.frame=frame;self.frames+=1
    vertices=[]
    for prim,pts in self.robot_meshes:
     matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
     world=np.c_[pts,np.ones(len(pts))]@matrix
     vertices.extend(world[:,:2])
    points=np.unique(np.asarray(vertices),axis=0);poly=points[ConvexHull(points).vertices]
    camera=UsdGeom.Camera(self.stage.GetPrimAtPath(self.robot_path+'/chassis_link/front_hawk/left/camera_left'))
    cam=np.asarray(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()));projection=np.asarray(camera.GetCamera(Usd.TimeCode.Default()).frustum.ComputeProjectionMatrix())
    observations=[]
    for skel,capsule in self.humans:
     center=np.asarray(capsule.GetPrim().GetAttribute('xformOp:translate').Get())
     bounds=np.array([center+[x,y,z] for x in [-RADIUS,RADIUS] for y in [-RADIUS,RADIUS] for z in [-HEIGHT/2,HEIGHT/2]])
     fov=intersects_crop(bounds,cam,projection);gap=polygon_circle_gap(poly,center[:2],RADIUS)
     observations.append({'person':str(skel.GetPath()),'in_camera_fov':fov,'surface_gap_m':gap,'center':center.tolist()})
    gaps=[h['surface_gap_m'] for h in observations if h['in_camera_fov']]
    if gaps:
     self.fov_frames+=1;current=min(gaps);self.min_clearance=current if self.min_clearance is None else min(current,self.min_clearance)
    self._fov_stream.write(json.dumps({'frame':frame,'people':observations})+'\n')
    if self.frames%60==0:self._fov_stream.flush()
    self.sync()
    self.measure_compute_seconds+=time.perf_counter()-measure_started
 def result(self):
    self._fov_stream.flush();self._contact_stream.flush()
    return {'measurement_version':'physx_capsules_fov_v1','contact_human_collision':self.human_collision,'contact_obstacle_collision':self.obstacle_collision,'contact_collision':self.human_collision or self.obstacle_collision,'contact_pairs':len(self.contact_pairs),'minimum_fov_person_surface_clearance_m':self.min_clearance,'fov_clearance_frames':self.fov_frames,'measurement_frames':self.frames,'metric_compute_seconds':self.measure_compute_seconds,'human_collision_body':{'shape':'kinematic_capsule','radius_m':RADIUS,'height_m':HEIGHT},'clearance_geometry':'planar convex hull of robot collision geometry (cylinders sampled at128 angles) to human capsule surface; capsule bounds intersect center-cropped camera FOV; no occlusion test'}
 def close(self):
    self.active=False;self.sub=None
    for stream in [self._fov_stream,self._contact_stream]:
     if not stream.closed:stream.close()
