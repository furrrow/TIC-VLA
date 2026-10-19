"""VEGA bridge retaining DynaNav's robot/controller/evaluator. No goal coordinates sent."""
import json, os, urllib.request, shutil, time
from pathlib import Path
import torch
class VegaClient:
    def __init__(self):
        self.url=os.environ['VEGA_POLICY_URL']
        self._reviewed=False
        from visible_obstacles import add_configured_obstacles
        add_configured_obstacles()
        with urllib.request.urlopen(self.url,timeout=10) as response:
            assert response.read()==b'ready'
    def reset_episode_state(self):
        pass
    def predict_async(self, *, image_paths, instruction, current_step, current_robot_pose, **kwargs):
        review_root=os.environ.get('VEGA_REVIEW_ROOT')
        if review_root and not self._reviewed:
            review=Path(review_root)/os.environ['TICVLA_EPISODE_NAME']
            review.mkdir(parents=True,exist_ok=True)
            shutil.copy2(image_paths[-1],review/'initial_camera.jpg')
            import omni.usd
            from pxr import Usd,UsdGeom
            stage=omni.usd.get_context().get_stage()
            human_states=[]
            parent=stage.GetPrimAtPath('/World/Characters')
            if parent and parent.IsValid():
                for child in parent.GetChildren():
                    if child.GetName()!='Character' and not child.GetName().startswith('Character_'):continue
                    prims=list(Usd.PrimRange(child))
                    skel=next((p for p in prims if p.GetTypeName()=='SkelRoot'),child)
                    pos=UsdGeom.Xformable(skel).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                    human_states.append({'prim':str(child.GetPath()),'position':list(pos),'physics_schemas':sorted({schema for p in prims for schema in p.GetAppliedSchemas() if 'Collision' in schema or 'CharacterController' in schema})})
            (review/'human_layout.json').write_text(json.dumps(human_states))
            (review/'observation.json').write_text(json.dumps({'instruction':instruction,'frame':current_step,'robot_pose':current_robot_pose}))
            print(f'VEGA_VISIBILITY_REVIEW_WAIT {review}',flush=True)
            deadline=time.monotonic()+1200
            while not (review/'approved.json').exists():
                if (review/'rejected.json').exists() or time.monotonic()>deadline:
                    raise RuntimeError('Visibility review rejected or timed out; exclude episode')
                time.sleep(1)
            self._reviewed=True
        def plain(value):
            if hasattr(value,'detach'):return value.detach().cpu().tolist()
            if hasattr(value,'tolist'):return value.tolist()
            if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [plain(v) for v in value]
            return value
        payload=json.dumps({'image_path':os.path.abspath(image_paths[-1]),'image_paths':[os.path.abspath(p) for p in image_paths],'instruction':instruction,'frame':current_step,'robot_pose':current_robot_pose,'episode':os.environ.get('TICVLA_EPISODE_NAME'),'run_id':os.environ.get('BENCHMARK_RUN_ID'),'robot_state':plain(kwargs.get('robot_state')),'time_delay':plain(kwargs.get('time_delay')),'previous_waypoints_text':plain(kwargs.get('previous_waypoints_text'))}).encode()
        request=urllib.request.Request(self.url,data=payload,headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=120) as response:
                result=json.load(response)
        except Exception as error:
            print(f"VEGA_INFERENCE_ERROR frame={current_step}: {error}",flush=True)
            raise
        print(f"VEGA_INFERENCE_OK frame={current_step} latency_s={result['latency_s']:.4f}",flush=True)
        self.native_control = result.get('velocity_command')
        from vega_trace import save_prediction
        save_prediction(image_paths[-1],instruction,current_step,current_robot_pose,result)
        waypoints=torch.tensor(result['waypoints'],dtype=torch.float32)[None]
        return '',waypoints,current_step,True,current_robot_pose
