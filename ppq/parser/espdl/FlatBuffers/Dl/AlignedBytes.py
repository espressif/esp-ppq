# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Dl

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class AlignedBytes(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        return 16

    # AlignedBytes
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # AlignedBytes
    def Bytes(self, j = None):
        if j is None:
            return [self._tab.Get(flatbuffers.number_types.Uint8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0 + i * 1)) for i in range(self.BytesLength())]
        elif j >= 0 and j < self.BytesLength():
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0 + j * 1))
        else:
            return None

    # AlignedBytes
    def BytesAsNumpy(self):
        return self._tab.GetArrayAsNumpy(flatbuffers.number_types.Uint8Flags, self._tab.Pos + 0, self.BytesLength())

    # AlignedBytes
    def BytesLength(self):
        return 16

    # AlignedBytes
    def BytesIsNone(self):
        return False


def CreateAlignedBytes(builder, bytes):
    builder.Prep(16, 16)
    for _idx0 in range(16 , 0, -1):
        builder.PrependUint8(bytes[_idx0-1])
    return builder.Offset()