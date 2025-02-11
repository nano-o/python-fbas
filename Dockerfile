FROM python:3.11.4

# Create virtual environment in a safer location
RUN python3 -m venv /app/venv
# RUN ln -s /opt/venv/bin/python3 /opt/venv/bin/python  # Ensure python exists

RUN useradd -ms /bin/bash admin

COPY . /app

WORKDIR /app

# Install dependencies inside venv
RUN . venv/bin/activate && yes | pip install .
RUN . venv/bin/activate && yes | pip install notebook

# Run make install inside venv
# RUN . venv/bin/activate && make install -C /app/python_fbas/constellation/brute-force-search
RUN . venv/bin/activate && cd python_fbas/constellation/brute-force-search && make install

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

USER admin
ENTRYPOINT ["/docker-entrypoint.sh"]

CMD jupyter notebook python_fbas/constellation/notebook.ipynb --ip 0.0.0.0 --no-browser --allow-root

# working version
# FROM python:3.11.4

# # Create virtual environment in a safer location
# RUN python3 -m venv /app/venv
# # RUN ln -s /opt/venv/bin/python3 /opt/venv/bin/python  # Ensure python exists

# COPY . /app

# WORKDIR /app

# # Install dependencies inside venv
# RUN . venv/bin/activate && yes | pip install .

# # Run make install inside venv
# # RUN . venv/bin/activate && make install -C /app/python_fbas/constellation/brute-force-search
# RUN . venv/bin/activate && cd python_fbas/constellation/brute-force-search && make install

# CMD . venv/bin/activate && cd venv/bin && ls
# CMD . venv/bin/activate && constellation compute-clusters --thresholds 100 130 100 140 --max-num-clusters=5 --min-cluster-size=33
