import socket
import json
import numpy as np
import struct
from threading import Thread
from .persistent_store import PersistentVectorStore
from .index_manager import IndexManager
from .query_engine import QueryEngine

class VectorDBServer:
    def __init__(self, host='127.0.0.1', port=6379, storage_path="vector_db"):
        self.host = host
        self.port = port
        self.store = PersistentVectorStore(storage_path)
        self.index_manager = IndexManager(self.store)
        self.query_engine = QueryEngine(self.index_manager)
        self.running = False
        
    def start(self):
        """Start the server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"Server running on {self.host}:{self.port}")
        
        try:
            while self.running:
                client_socket, address = self.server_socket.accept()
                print(f"Connection from {address}")
                client_thread = Thread(target=self.handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
        finally:
            self.server_socket.close()
            
    def stop(self):
        """Stop the server."""
        self.running = False
        self.server_socket.close()
        
    def handle_client(self, client_socket):
        """Handle client connections."""
        try:
            while True:
                # Receive message length first
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                    
                msg_length = struct.unpack('>I', length_data)[0]
                
                # Receive the full message in chunks
                chunks = []
                bytes_received = 0
                while bytes_received < msg_length:
                    chunk = client_socket.recv(min(msg_length - bytes_received, 8192))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    bytes_received += len(chunk)
                    
                if not chunks:
                    break
                    
                data = b''.join(chunks)
                
                try:
                    request = json.loads(data.decode('utf-8'))
                    response = self.handle_request(request)
                    
                    # Send response with length prefix
                    response_data = json.dumps(response).encode('utf-8')
                    client_socket.send(struct.pack('>I', len(response_data)))
                    client_socket.send(response_data)
                    
                except Exception as e:
                    error_response = {'status': 'error', 'message': str(e)}
                    response_data = json.dumps(error_response).encode('utf-8')
                    client_socket.send(struct.pack('>I', len(response_data)))
                    client_socket.send(response_data)
        finally:
            client_socket.close()
            
    def handle_request(self, request):
        """Handle different types of requests."""
        command = request.get('command')
        
        if command == 'add':
            self.store.add(request['id'], np.array(request['vector']))
            return {'status': 'success', 'message': 'Vector added'}
            
        elif command == 'get':
            vector = self.store.get(request['id'])
            if vector is not None:
                return {'status': 'success', 'vector': vector.tolist()}
            return {'status': 'error', 'message': 'Vector not found'}
            
        elif command == 'remove':
            success = self.store.remove(request['id'])
            if success:
                return {'status': 'success', 'message': 'Vector removed'}
            return {'status': 'error', 'message': 'Vector not found'}
            
        elif command == 'search':
            self.index_manager.update_index()
            results = self.query_engine.search(
                np.array(request['query_vector']),
                k=request.get('k', 10),
                metric=request.get('metric', 'cosine')
            )
            return {'status': 'success', 'results': [(str(id), float(score)) for id, score in results]}
            
        elif command == 'clear':
            self.store.clear()
            return {'status': 'success', 'message': 'Database cleared'}
            
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'} 