import uvicorn
from api.server import app


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1, # Set to 1 for development, increase for production
        reload=True, # Only for development, remove in production
        # Trust the forwarded headers set by the reverse proxy in
        # deploy/nginx, so the app sees the real client address and scheme.
        # Only 127.0.0.1's headers are trusted -- i.e. a proxy on this host.
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )
