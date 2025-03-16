import socket
import json
import numpy as np
import struct

class VectorDBClient:
    def __init__(self, host='127.0.0.1', port=6379):
        self.host = host
        self.port = port
        
    def _send_request(self, request):
        """Send a request to the server and get the response."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            
            # Convert request to JSON and encode
            data = json.dumps(request).encode('utf-8')
            
            # Send message length first
            sock.send(struct.pack('>I', len(data)))
            
            # Send the data
            sock.send(data)
            
            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                return {'status': 'error', 'message': 'Connection closed'}
                
            msg_length = struct.unpack('>I', length_data)[0]
            
            # Receive the response in chunks
            chunks = []
            bytes_received = 0
            while bytes_received < msg_length:
                chunk = sock.recv(min(msg_length - bytes_received, 8192))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_received += len(chunk)
                
            response_data = b''.join(chunks)
            return json.loads(response_data.decode('utf-8'))
            
    def add_vector(self, id, vector):
        """Add a vector to the database."""
        request = {
            'command': 'add',
            'id': id,
            'vector': vector.tolist() if isinstance(vector, np.ndarray) else vector
        }
        return self._send_request(request)
        
    def get_vector(self, id):
        """Retrieve a vector by ID."""
        request = {
            'command': 'get',
            'id': id
        }
        response = self._send_request(request)
        if response['status'] == 'success':
            return np.array(response['vector'])
        print(f"Error getting vector: {response['message']}")  # Add error logging
        return None
        
    def remove_vector(self, id):
        """Remove a vector from the database."""
        request = {
            'command': 'remove',
            'id': id
        }
        return self._send_request(request)
        
    def search_similar(self, query_vector, k=10, metric='cosine'):
        """Search for similar vectors."""
        request = {
            'command': 'search',
            'query_vector': query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            'k': k,
            'metric': metric
        }
        return self._send_request(request)
        
    def clear_database(self):
        """Clear all vectors from the database."""
        request = {
            'command': 'clear'
        }
        return self._send_request(request) 