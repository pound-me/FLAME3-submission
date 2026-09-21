import unittest
from wddm_admission import blockers


class AdmissionTests(unittest.TestCase):
    def sample(self, *, processes=None, engines=None, utilization=1, memory=2000):
        return dict(processes=processes or {},engines=engines or [],
                    utilization_percent=utilization,memory_used_mib=memory)

    def test_idle_matlab_context_is_not_active_compute(self):
        self.assertFalse(blockers(self.sample(processes={4:'MATLAB.exe'}),own_pid=9))

    def test_active_unknown_blocks(self):
        self.assertTrue(blockers(self.sample(engines=[dict(pid=4,percent=.051)]),own_pid=9))

    def test_unicode_process_path_is_supported(self):
        observation=self.sample(processes={4:r'D:\????\SunloginClient.exe'},
                                engines=[dict(pid=4,percent=.051)])
        self.assertTrue(blockers(observation,own_pid=9))

    def test_other_python_blocks_even_if_idle(self):
        self.assertTrue(blockers(self.sample(processes={4:r'C:\python.exe'}),own_pid=9))

    def test_own_compute_not_external(self):
        self.assertFalse(blockers(self.sample(processes={9:r'C:\python.exe'},
                         engines=[dict(pid=9,percent=99)],utilization=99,memory=10000),
                         own_pid=9,timing=True))

    def test_normal_desktop_tiny_usage_allowed(self):
        self.assertFalse(blockers(self.sample(processes={4:r'C:\dwm.exe'},
                         engines=[dict(pid=4,percent=.1)]),own_pid=9,timing=True))

    def test_busy_desktop_invalidates_timing(self):
        self.assertTrue(blockers(self.sample(processes={4:r'C:\dwm.exe'},
                        engines=[dict(pid=4,percent=.51)]),own_pid=9,timing=True))

    def test_overall_busy_blocks_training(self):
        self.assertTrue(blockers(self.sample(utilization=6),own_pid=9))
        self.assertTrue(blockers(self.sample(memory=4097),own_pid=9))


if __name__=='__main__':
    unittest.main()
