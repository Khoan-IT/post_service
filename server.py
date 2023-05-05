import os
import logging
import grpc
import argparse

import post_slot_service_pb2
import post_slot_service_pb2_grpc

from pipeline import PostSlotModel
from concurrent import futures

post_slot_interface = os.environ.get('POST_SLOT_SERVER_INTERFACE', '0.0.0.0')
post_slot_port = int(os.environ.get('POST_SLOT_SERVER_PORT', 5002))
post_slot_model_path = os.environ.get('POST_SLOT_MODEL_PATH', 'model')

class PSServiceServicer(post_slot_service_pb2_grpc.PSServiceServicer):
    
    def __init__(self, model_path):
        self.recognizer = PostSlotModel(model_path)
        self.model_version = open(os.path.join(model_path, 'model_version.txt'), 'r', encoding='utf-8').read()
        
        logging.info(f"Loaded model {self.model_version}!")
        
    def PostSlotRecognize(self, request, context):
        result = self.recognizer(request.message)
        return post_slot_service_pb2.PostSlotRecognizeResponse(message=result)
    
def serve():
    logging.info("Server starting ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    post_slot_service_pb2_grpc.add_PSServiceServicer_to_server(
        PSServiceServicer(model_path=post_slot_model_path),
        server
    )
    server.add_insecure_port('{}:{}'.format(post_slot_interface, post_slot_port))
    server.start()
    logging.info(f"Started server on {post_slot_interface}:{post_slot_port}")
    server.wait_for_termination()
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    serve()
    