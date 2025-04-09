import os
import logging
import time
import uuid
from datetime import datetime
from fastapi import Request


# Setup logging
def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"api_{datetime.now().strftime('%Y%m%d')}.log")

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger("insurance-api")
    return logger


logger = setup_logger()


# Request tracking middleware
async def log_requests_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request_id to request state for access in route handlers
    request.state.request_id = request_id

    # Log request
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        # Log response
        logger.info(
            f"Request {request_id} completed: Status {response.status_code}, Time {process_time:.3f}s"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed: {str(e)}, Time {process_time:.3f}s")
        raise
