cd ~/mlbot2
pytest --junitxml=pytest.xml --cov-report=term-missing --cov=src ./test/ | tee pytest-coverage.txt