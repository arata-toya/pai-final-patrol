from pathlib import Path
import sys,unittest,json
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from control import PatrolControl,Telemetry

def item(label,gap=2.6):return SimpleNamespace(label=label,forward_m=gap)

class ControlTests(unittest.TestCase):
    def make(self):
        c=PatrolControl();t=Telemetry(0,0,0,0);c.start(0,t);return c,t
    def test_crack_can_stop_before_x5(self):
        c,t=self.make()
        for i in range(1,4):c.tick(i*.1,t,[item('crack')],i)
        self.assertEqual(c.state,'BRAKING')
    def test_stain_and_normal_do_not_trigger_inspection(self):
        c,t=self.make()
        for i in range(10):c.tick(i*.1,t,[item('stain'),item('normal')],i)
        self.assertEqual(c.state,'RUN');self.assertEqual(c.inspections,[])
        self.assertTrue(c.normal_seen);self.assertTrue(c.stain_seen)
    def test_uncertain_is_not_reported_as_crack(self):
        c,t=self.make()
        for i in range(10):c.tick(i*.1,t,[item('uncertain')],i)
        self.assertEqual(c.inspections,[]);self.assertEqual(c.state,'REVIEW_BRAKING')
    def test_distinct_camera_frames(self):
        c,t=self.make()
        for i in range(10):c.tick(i*.1,t,[item('crack')],1)
        self.assertEqual(c.state,'RUN')
    def test_distinct_odom_and_three_second_pause(self):
        c,t=self.make()
        for i in range(1,4):c.tick(i*.1,t,[item('crack')],i)
        for i in range(5):c.tick(.5+i*.01,Telemetry(.02,0,0,1))
        self.assertEqual(c.state,'BRAKING')
        c.tick(.7,Telemetry(.02,0,0,2));self.assertEqual(c.state,'INSPECT')
        c.tick(3.6,Telemetry(.02,0,0,3));self.assertEqual(c.state,'INSPECT')
        c.tick(3.8,Telemetry(.02,0,0,4));self.assertEqual(c.state,'EXIT')
    def test_stale_data_sends_zero(self):
        c,t=self.make();self.assertEqual(c.tick(.3,t,fresh=False),0);self.assertEqual(c.state,'FAULT')
    def test_no_crack_finishes_without_inspection(self):
        c,t=self.make();c.tick(1,Telemetry(12.3,.2,0,1),[],1)
        c.tick(1.1,Telemetry(12.31,0,0,2));c.tick(1.2,Telemetry(12.31,0,0,3))
        self.assertEqual(c.state,'DONE');self.assertEqual(c.inspections,[])
    def test_training_validation_test_no_duplicate_images(self):
        root=Path(__file__).resolve().parents[1]
        rows=json.loads((root/'data/manifest.json').read_text())
        groups={split:{r['sha256'] for r in rows if r['split']==split}
                for split in ['train','validation','test']}
        self.assertEqual([len(groups[s]) for s in groups],[720,180,180])
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

if __name__=='__main__':unittest.main()
