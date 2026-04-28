# Use official Python 3.11 slim runtime (stable and lightweight)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application scripts and preprocessed data
# Note: Ensure you have run preprocess_data.py first to generate preprocessed_data.csv
COPY generate_report.py /app/
COPY preprocessed_data.csv /app/

# Declare /app/output as a volume mount point so output saves to the host machine.
# When running: docker run -v "$(pwd)/output:/app/output" urban-pulse-nyc
VOLUME ["/app/output"]

# Create output directory inside the container
RUN mkdir -p /app/output

# Run the generation script when the container launches.
CMD ["python", "./generate_report.py"]
