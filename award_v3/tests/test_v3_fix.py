from pathlib import Path
import ast,datetime,json,sys,tempfile,time,unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from control import PatrolControl,Telemetry
ROOT=Path(__file__).resolve().parents[1]

class FixTests(unittest.TestCase):
    def test_uncertain_nearby_stops_for_review_not_crack(self):
        c=PatrolControl();t=Telemetry(0,0,0,0);c.start(0,t)
        d=SimpleNamespace(label='uncertain',forward_m=3.)
        for i in range(1,4):self.assertEqual(c.tick(i*.1,t,[d],i),0 if i==3 else .28)
        self.assertEqual(c.state,'REVIEW_BRAKING')
        c.tick(.4,Telemetry(0,0,0,1));c.tick(.5,Telemetry(0,0,0,2))
        self.assertEqual(c.state,'REVIEW_REQUIRED');self.assertEqual(c.inspections,[])
    def test_repeated_frame_not_enough_for_review(self):
        c=PatrolControl();t=Telemetry(0,0,0,0);c.start(0,t)
        for i in range(10):c.tick(i*.1,t,[SimpleNamespace(label='uncertain',forward_m=3.)],1)
        self.assertEqual(c.state,'RUN')
    def test_capture_mode_photographs_normal_without_ai_event(self):
        c=PatrolControl(capture_only=True);t=Telemetry(0,0,0,0);c.start(0,t)
        for i in range(1,4):c.tick(i*.1,t,[SimpleNamespace(label='normal',forward_m=2.7)],i)
        c.tick(.4,Telemetry(0,0,0,1));c.tick(.5,Telemetry(0,0,0,2))
        self.assertEqual(c.state,'INSPECT');self.assertEqual(len(c.inspections),1)
        events=[e['event'] for e in c.events]
        self.assertIn('ROI_BRAKING',events);self.assertIn('CAPTURE_CONFIRMED',events)
        self.assertNotIn('AI_BRAKING',events);self.assertNotIn('STOP_CONFIRMED',events)
    def test_confirmed_crack_keeps_original_path(self):
        c=PatrolControl();t=Telemetry(0,0,0,0);c.start(0,t)
        for i in range(1,4):c.tick(i*.1,t,[SimpleNamespace(label='crack',forward_m=2.7)],i)
        self.assertEqual(c.state,'BRAKING')
    def test_capture_node_paths_and_reports_without_ros(self):
        class FakeNode:
            def __init__(self,*a,**k):pass
            def create_subscription(self,*a,**k):return None
            def create_publisher(self,*a,**k):return None
        dummy=lambda *a,**k:None
        tree=ast.parse((ROOT/'scripts/patrol_v3.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        with tempfile.TemporaryDirectory() as temp:
            namespace=dict(Node=FakeNode,WallVision=dummy,PatrolControl=PatrolControl,
                           ROOT=Path(temp),time=time,Telemetry=Telemetry,CvBridge=dummy,
                           datetime=datetime.datetime,QoSProfile=dummy,json=json,
                           HistoryPolicy=SimpleNamespace(KEEP_LAST=1),
                           ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1),
                           Image=None,Odometry=None,Twist=None)
            import csv
            namespace['csv']=csv
            exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),
                         '<node-construction-test>','exec'),namespace)
            args=SimpleNamespace(capture_only=True,mode='cnn',scenario='CASE_A',stop_m=2.8,end_x=12.3,headless=True)
            node=namespace['PatrolNode'](args)
            self.assertTrue(node.output.name.endswith('_CASE_A_capture'))
            node.report_items=[dict(index=1,stop_x_m=.1,estimated_target_x_m=4.,confidence=.6,
                                   annotated_image='inspection_01_annotated.png')]
            node.write_report()
            report=(node.output/'inspection_report.html').read_text()
            self.assertIn('撮影データ（認識調整用）',report)
            self.assertNotIn('ひび割れ候補 1',report)
            node.trace_file.close();node.perception_file.close()

if __name__=='__main__':unittest.main()
