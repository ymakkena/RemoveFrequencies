class DepthException(Exception):
    pass


class Sound(object):
    def __init__(self, fs, snd):

        self.samples, self.channels = snd.shape
        self.fs = fs # Sample frequency
        self.duration = self.samples/fs
        self.snd = snd

        self.__check_depth(snd.dtype)

    def __check_depth(self, depth_type):
        # The data type used to store the samples

        # Produces interesting results
        # if depth_type.num == 2:
        #    self.depth = 8
        if depth_type.num == 3:
            self.depth = 16
        # Not supported by scipy
        # elif depth_type == 4:
        #     self.depth = 24
        elif depth_type.num == 5:
            self.depth = 32
        else:
            raise DepthException("Depth '{}' is unsupported".format(depth_type.name))
