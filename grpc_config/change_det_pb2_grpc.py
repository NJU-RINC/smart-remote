# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import grpc_config.change_det_pb2 as change__det__pb2


class DetStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Detect = channel.unary_unary(
                '/det.Det/Detect',
                request_serializer=change__det__pb2.DetInput.SerializeToString,
                response_deserializer=change__det__pb2.DetResult.FromString,
                )


class DetServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Detect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DetServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Detect': grpc.unary_unary_rpc_method_handler(
                    servicer.Detect,
                    request_deserializer=change__det__pb2.DetInput.FromString,
                    response_serializer=change__det__pb2.DetResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'det.Det', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Det(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Detect(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/det.Det/Detect',
            change__det__pb2.DetInput.SerializeToString,
            change__det__pb2.DetResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)