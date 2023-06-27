from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DetInput(_message.Message):
    __slots__ = ["base", "target"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    base: bytes
    target: bytes
    def __init__(self, target: _Optional[bytes] = ..., base: _Optional[bytes] = ...) -> None: ...

class DetResult(_message.Message):
    __slots__ = ["boxes"]
    BOXES_FIELD_NUMBER: _ClassVar[int]
    boxes: _containers.RepeatedCompositeFieldContainer[Rect]
    def __init__(self, boxes: _Optional[_Iterable[_Union[Rect, _Mapping]]] = ...) -> None: ...

class Rect(_message.Message):
    __slots__ = ["bottom", "label", "left", "logit", "right", "top"]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    LOGIT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    bottom: float
    label: int
    left: float
    logit: float
    right: float
    top: float
    def __init__(self, left: _Optional[float] = ..., top: _Optional[float] = ..., right: _Optional[float] = ..., bottom: _Optional[float] = ..., label: _Optional[int] = ..., logit: _Optional[float] = ...) -> None: ...
