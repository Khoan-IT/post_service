import os
import logging
import grpc
import argparse

import intent_slot_service_pb2
import intent_slot_service_pb2_grpc

from pipeline import IntentSlotModel
from concurrent import futures

intent_slot_interface = os.environ.get('INTENT_SLOT_SERVER_INTERFACE', '0.0.0.0')
intent_slot_port = int(os.environ.get('INTENT_SLOT_SERVER_PORT', 5002))
intent_slot_model_path = os.environ.get('INTENT_SLOT_MODEL_PATH', 'model')

class ISServiceServicer(intent_slot_service_pb2_grpc.ISServiceServicer):
    
    def __init__(self, model_path):
        self.recognizer = IntentSlotModel(model_path)
        self.model_version = open(os.path.join(model_path, 'model_version.txt'), 'r', encoding='utf-8').read()
        
        logging.info(f"Loaded model {self.model_version}!")
        
    def IntentSlotRecognize(self, request, context):
        result = self.recognizer(request.message)
        return intent_slot_service_pb2.IntentSlotRecognizeResponse(message=result)
    
def serve():
    logging.info("Server starting ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    intent_slot_service_pb2_grpc.add_ISServiceServicer_to_server(
        ISServiceServicer(model_path=intent_slot_model_path),
        server
    )
    server.add_insecure_port('{}:{}'.format(intent_slot_interface, intent_slot_port))
    server.start()
    logging.info(f"Started server on {intent_slot_interface}:{intent_slot_port}")
    server.wait_for_termination()
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    serve()
    