import sys

def pytest_sessionstart(session):
    sys.path.append("my_app")