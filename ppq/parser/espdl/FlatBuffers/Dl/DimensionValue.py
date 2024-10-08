# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Dl

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DimensionValue(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DimensionValue()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDimensionValue(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # DimensionValue
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DimensionValue
    def DimType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DimensionValue
    def DimValue(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # DimensionValue
    def DimParam(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def DimensionValueStart(builder):
    builder.StartObject(3)

def Start(builder):
    DimensionValueStart(builder)

def DimensionValueAddDimType(builder, dimType):
    builder.PrependInt8Slot(0, dimType, 0)

def AddDimType(builder, dimType):
    DimensionValueAddDimType(builder, dimType)

def DimensionValueAddDimValue(builder, dimValue):
    builder.PrependInt64Slot(1, dimValue, 0)

def AddDimValue(builder, dimValue):
    DimensionValueAddDimValue(builder, dimValue)

def DimensionValueAddDimParam(builder, dimParam):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dimParam), 0)

def AddDimParam(builder, dimParam):
    DimensionValueAddDimParam(builder, dimParam)

def DimensionValueEnd(builder):
    return builder.EndObject()

def End(builder):
    return DimensionValueEnd(builder)


class DimensionValueT(object):

    # DimensionValueT
    def __init__(self):
        self.dimType = 0  # type: int
        self.dimValue = 0  # type: int
        self.dimParam = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dimensionValue = DimensionValue()
        dimensionValue.Init(buf, pos)
        return cls.InitFromObj(dimensionValue)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, dimensionValue):
        x = DimensionValueT()
        x._UnPack(dimensionValue)
        return x

    # DimensionValueT
    def _UnPack(self, dimensionValue):
        if dimensionValue is None:
            return
        self.dimType = dimensionValue.DimType()
        self.dimValue = dimensionValue.DimValue()
        self.dimParam = dimensionValue.DimParam()

    # DimensionValueT
    def Pack(self, builder):
        if self.dimParam is not None:
            dimParam = builder.CreateString(self.dimParam)
        DimensionValueStart(builder)
        DimensionValueAddDimType(builder, self.dimType)
        DimensionValueAddDimValue(builder, self.dimValue)
        if self.dimParam is not None:
            DimensionValueAddDimParam(builder, dimParam)
        dimensionValue = DimensionValueEnd(builder)
        return dimensionValue
