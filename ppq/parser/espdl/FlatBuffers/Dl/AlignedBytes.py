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

try:
    from typing import List
except:
    pass

class AlignedBytesT(object):

    # AlignedBytesT
    def __init__(self):
        self.bytes = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        alignedBytes = AlignedBytes()
        alignedBytes.Init(buf, pos)
        return cls.InitFromObj(alignedBytes)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, alignedBytes):
        x = AlignedBytesT()
        x._UnPack(alignedBytes)
        return x

    # AlignedBytesT
    def _UnPack(self, alignedBytes):
        if alignedBytes is None:
            return
        if not alignedBytes.BytesIsNone():
            if np is None:
                self.bytes = []
                for i in range(alignedBytes.BytesLength()):
                    self.bytes.append(alignedBytes.Bytes(i))
            else:
                self.bytes = alignedBytes.BytesAsNumpy()

    # AlignedBytesT
    def Pack(self, builder):
        return CreateAlignedBytes(builder, self.bytes)
