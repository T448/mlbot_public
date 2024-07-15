cd ~/mlbot_public
pytest --junitxml=pytest.xml --cov-report=term-missing --cov=src ./test/ | tee pytest-coverage.txt