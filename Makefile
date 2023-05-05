GRPC_SOURCES = intent_slot_service_pb2.py intent_slot_service_pb2_grpc.py

all: $(GRPC_SOURCES)

$(GRPC_SOURCES): intent_slot_service.proto
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. intent_slot_service.proto

clean:
	rm $(GRPC_SOURCES)