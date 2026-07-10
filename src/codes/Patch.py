class PatchTwoD:
    def __init__(self, x=0., z=5000., dd_width=10000., dip=90.):
        self.x = x
        self.z = z
        self.dd_width = dd_width
        self.dip = dip
    
    @property
    def top(self):
        return self.z - self.dd_width/2
    
    @property
    def bottom(self):
        return self.z + self.dd_width/2