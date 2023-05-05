import os
import sys
import json
import copy
import time

import grpc
import post_slot_service_pb2
import post_slot_service_pb2_grpc

from underthesea import word_tokenize
from normalizer import V2PostNormalizer
from result.elastic_formatter import PostFormatter

MONITOR_SERVER_INTERFACE = os.environ.get('HOST', 'localhost')
MONITOR_SERVER_PORT = int(os.environ.get('PORT', 5002))

CHANNEL_IP = f"{MONITOR_SERVER_INTERFACE}:{MONITOR_SERVER_PORT}"

def main():
    channel = grpc.insecure_channel(CHANNEL_IP)
    stub = post_slot_service_pb2_grpc.PSServiceStub(channel)
    
    normalizer = V2PostNormalizer()
    es_formatter = PostFormatter()
    
    with open('./result/original/v2_raw_post_model.json', 'r') as f:
        posts = json.load(f)
        message = []
        for post in posts:
            temp_post = {"id": post["_id"]["$oid"]}
            temp_post['content'] = " ".join(normalizer.v2_normalize(post['content']))
            message.append(temp_post)
            break
        message = json.dumps(message)
        result = stub.PostSlotRecognize(post_slot_service_pb2.PostSlotRecognizeRequest(message=message))
        print(es_formatter.get_activities(json.loads(result.message)))
if __name__ == "__main__":
    main()