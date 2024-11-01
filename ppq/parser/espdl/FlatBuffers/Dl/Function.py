# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Dl

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Function(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Function()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFunction(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Function
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Function
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Function
    def Input(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Function
    def InputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Function
    def InputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Function
    def Output(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Function
    def OutputLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Function
    def OutputIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Function
    def Attribute(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Function
    def AttributeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Function
    def AttributeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # Function
    def AttributeProto(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.Attribute import Attribute
            obj = Attribute()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Function
    def AttributeProtoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Function
    def AttributeProtoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Function
    def Node(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.Node import Node
            obj = Node()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Function
    def NodeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Function
    def NodeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Function
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Function
    def OpsetImport(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlatBuffers.Dl.OperatorSetId import OperatorSetId
            obj = OperatorSetId()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Function
    def OpsetImportLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Function
    def OpsetImportIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # Function
    def Domain(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def FunctionStart(builder):
    builder.StartObject(9)

def Start(builder):
    FunctionStart(builder)

def FunctionAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    FunctionAddName(builder, name)

def FunctionAddInput(builder, input):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(input), 0)

def AddInput(builder, input):
    FunctionAddInput(builder, input)

def FunctionStartInputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputVector(builder, numElems):
    return FunctionStartInputVector(builder, numElems)

def FunctionAddOutput(builder, output):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(output), 0)

def AddOutput(builder, output):
    FunctionAddOutput(builder, output)

def FunctionStartOutputVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputVector(builder, numElems):
    return FunctionStartOutputVector(builder, numElems)

def FunctionAddAttribute(builder, attribute):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(attribute), 0)

def AddAttribute(builder, attribute):
    FunctionAddAttribute(builder, attribute)

def FunctionStartAttributeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartAttributeVector(builder, numElems):
    return FunctionStartAttributeVector(builder, numElems)

def FunctionAddAttributeProto(builder, attributeProto):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(attributeProto), 0)

def AddAttributeProto(builder, attributeProto):
    FunctionAddAttributeProto(builder, attributeProto)

def FunctionStartAttributeProtoVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartAttributeProtoVector(builder, numElems):
    return FunctionStartAttributeProtoVector(builder, numElems)

def FunctionAddNode(builder, node):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(node), 0)

def AddNode(builder, node):
    FunctionAddNode(builder, node)

def FunctionStartNodeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeVector(builder, numElems):
    return FunctionStartNodeVector(builder, numElems)

def FunctionAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    FunctionAddDocString(builder, docString)

def FunctionAddOpsetImport(builder, opsetImport):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(opsetImport), 0)

def AddOpsetImport(builder, opsetImport):
    FunctionAddOpsetImport(builder, opsetImport)

def FunctionStartOpsetImportVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOpsetImportVector(builder, numElems):
    return FunctionStartOpsetImportVector(builder, numElems)

def FunctionAddDomain(builder, domain):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(domain), 0)

def AddDomain(builder, domain):
    FunctionAddDomain(builder, domain)

def FunctionEnd(builder):
    return builder.EndObject()

def End(builder):
    return FunctionEnd(builder)

import FlatBuffers.Dl.Attribute
import FlatBuffers.Dl.Node
import FlatBuffers.Dl.OperatorSetId
try:
    from typing import List
except:
    pass

class FunctionT(object):

    # FunctionT
    def __init__(self):
        self.name = None  # type: str
        self.input = None  # type: List[str]
        self.output = None  # type: List[str]
        self.attribute = None  # type: List[str]
        self.attributeProto = None  # type: List[FlatBuffers.Dl.Attribute.AttributeT]
        self.node = None  # type: List[FlatBuffers.Dl.Node.NodeT]
        self.docString = None  # type: str
        self.opsetImport = None  # type: List[FlatBuffers.Dl.OperatorSetId.OperatorSetIdT]
        self.domain = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        function = Function()
        function.Init(buf, pos)
        return cls.InitFromObj(function)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, function):
        x = FunctionT()
        x._UnPack(function)
        return x

    # FunctionT
    def _UnPack(self, function):
        if function is None:
            return
        self.name = function.Name()
        if not function.InputIsNone():
            self.input = []
            for i in range(function.InputLength()):
                self.input.append(function.Input(i))
        if not function.OutputIsNone():
            self.output = []
            for i in range(function.OutputLength()):
                self.output.append(function.Output(i))
        if not function.AttributeIsNone():
            self.attribute = []
            for i in range(function.AttributeLength()):
                self.attribute.append(function.Attribute(i))
        if not function.AttributeProtoIsNone():
            self.attributeProto = []
            for i in range(function.AttributeProtoLength()):
                if function.AttributeProto(i) is None:
                    self.attributeProto.append(None)
                else:
                    attribute_ = FlatBuffers.Dl.Attribute.AttributeT.InitFromObj(function.AttributeProto(i))
                    self.attributeProto.append(attribute_)
        if not function.NodeIsNone():
            self.node = []
            for i in range(function.NodeLength()):
                if function.Node(i) is None:
                    self.node.append(None)
                else:
                    node_ = FlatBuffers.Dl.Node.NodeT.InitFromObj(function.Node(i))
                    self.node.append(node_)
        self.docString = function.DocString()
        if not function.OpsetImportIsNone():
            self.opsetImport = []
            for i in range(function.OpsetImportLength()):
                if function.OpsetImport(i) is None:
                    self.opsetImport.append(None)
                else:
                    operatorSetId_ = FlatBuffers.Dl.OperatorSetId.OperatorSetIdT.InitFromObj(function.OpsetImport(i))
                    self.opsetImport.append(operatorSetId_)
        self.domain = function.Domain()

    # FunctionT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.input is not None:
            inputlist = []
            for i in range(len(self.input)):
                inputlist.append(builder.CreateString(self.input[i]))
            FunctionStartInputVector(builder, len(self.input))
            for i in reversed(range(len(self.input))):
                builder.PrependUOffsetTRelative(inputlist[i])
            input = builder.EndVector()
        if self.output is not None:
            outputlist = []
            for i in range(len(self.output)):
                outputlist.append(builder.CreateString(self.output[i]))
            FunctionStartOutputVector(builder, len(self.output))
            for i in reversed(range(len(self.output))):
                builder.PrependUOffsetTRelative(outputlist[i])
            output = builder.EndVector()
        if self.attribute is not None:
            attributelist = []
            for i in range(len(self.attribute)):
                attributelist.append(builder.CreateString(self.attribute[i]))
            FunctionStartAttributeVector(builder, len(self.attribute))
            for i in reversed(range(len(self.attribute))):
                builder.PrependUOffsetTRelative(attributelist[i])
            attribute = builder.EndVector()
        if self.attributeProto is not None:
            attributeProtolist = []
            for i in range(len(self.attributeProto)):
                attributeProtolist.append(self.attributeProto[i].Pack(builder))
            FunctionStartAttributeProtoVector(builder, len(self.attributeProto))
            for i in reversed(range(len(self.attributeProto))):
                builder.PrependUOffsetTRelative(attributeProtolist[i])
            attributeProto = builder.EndVector()
        if self.node is not None:
            nodelist = []
            for i in range(len(self.node)):
                nodelist.append(self.node[i].Pack(builder))
            FunctionStartNodeVector(builder, len(self.node))
            for i in reversed(range(len(self.node))):
                builder.PrependUOffsetTRelative(nodelist[i])
            node = builder.EndVector()
        if self.docString is not None:
            docString = builder.CreateString(self.docString)
        if self.opsetImport is not None:
            opsetImportlist = []
            for i in range(len(self.opsetImport)):
                opsetImportlist.append(self.opsetImport[i].Pack(builder))
            FunctionStartOpsetImportVector(builder, len(self.opsetImport))
            for i in reversed(range(len(self.opsetImport))):
                builder.PrependUOffsetTRelative(opsetImportlist[i])
            opsetImport = builder.EndVector()
        if self.domain is not None:
            domain = builder.CreateString(self.domain)
        FunctionStart(builder)
        if self.name is not None:
            FunctionAddName(builder, name)
        if self.input is not None:
            FunctionAddInput(builder, input)
        if self.output is not None:
            FunctionAddOutput(builder, output)
        if self.attribute is not None:
            FunctionAddAttribute(builder, attribute)
        if self.attributeProto is not None:
            FunctionAddAttributeProto(builder, attributeProto)
        if self.node is not None:
            FunctionAddNode(builder, node)
        if self.docString is not None:
            FunctionAddDocString(builder, docString)
        if self.opsetImport is not None:
            FunctionAddOpsetImport(builder, opsetImport)
        if self.domain is not None:
            FunctionAddDomain(builder, domain)
        function = FunctionEnd(builder)
        return function