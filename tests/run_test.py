import sys, os
sys.path.append('/mnt/d/Python/minifympi')

from minifympi.tests.test_send_recv import *
from minifympi.tests.test_exec import *

import unittest

if __name__ == '__main__':
    unittest.main()