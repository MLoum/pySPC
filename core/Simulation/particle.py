

class Particle:
    def __init__(self):
        self.pos = None
        self.angle = None

        self.type = "spherical"

        self.diff_trans = None
        self.diff_rot = None

    def move(self):
        pass

    def light_interaction(self, electrical_field):
        pass


