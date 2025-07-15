import pytest
import logging

@pytest.fixture(autouse=True)
def log_test_start_and_end(request):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"START TEST: {request.node.name}")
    yield
    logging.info(f"END TEST: {request.node.name}")
