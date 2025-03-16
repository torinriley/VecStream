#!/usr/bin/env python3
import argparse
from vecstream.server import VectorDBServer

def main():
    parser = argparse.ArgumentParser(description='VectorDB Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host address to bind to')
    parser.add_argument('--port', type=int, default=6379, help='Port to listen on')
    parser.add_argument('--storage', default='vector_db', help='Path to store vector database files')
    
    args = parser.parse_args()
    
    server = VectorDBServer(
        host=args.host,
        port=args.port,
        storage_path=args.storage
    )
    
    try:
        print(f"Starting VectorDB server on {args.host}:{args.port}")
        print(f"Storage path: {args.storage}")
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.stop()

if __name__ == '__main__':
    main() 