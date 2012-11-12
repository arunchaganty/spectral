"""Progress Bar"""

import time
import sys

class ProgressBar:
    """Progress Bar"""

    def __init__(self, width = 40):
        """Initialise with some width"""
        self.width = width
        self.state = 0
        self.total = 0

    def start(self, total_iterations):
        """Set up a scaling factor for total iterations"""
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        # return to start of line, after '['
        sys.stdout.write("\b" * (self.width+1)) 

        self.state, self.total = 0, total_iterations

    def update(self, n):
        """Update the ticker"""
        state_ = int(self.width*n)/int(self.total)
        if state_ == self.state:
            pass
        elif self.state < self.width and state_ > self.state:
            # update the bar
            for i in xrange( state_ - self.state ):
                sys.stdout.write("-")
            sys.stdout.flush()
            self.state = state_

    def stop(self):
        """End the ticker"""
        for i in xrange( self.width - self.state ):
            sys.stdout.write("-")
        sys.stdout.write("\n")
        self.state, self.total = 0, 0

